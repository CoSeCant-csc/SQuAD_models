#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of module layers/NN modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------



class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)   # len * batch * hdim

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]  # len * batch * hdim
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()  # batch
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)  # index of batch with length from big to small
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)  # len * batch * hdim

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]  # len * batch * hdim

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class BoundaryPointerLayer(nn.Module):
    """
    F_k = tanh(VH^r + (W^a h_{k-1}^a + b^a) \otimes e_{(P+1)})
    \beta_k = softmax(v^T F_k + c\otimes e_{(P+1)})
    h_k^a = LSTM(H^r}\beta_k^T,h_{k-1}^a)

    Probility of word j is k-th word in answer
    p(a_k=j|a_1,a_2,\dots,a_{k-1}, H^r) = \beta_{k,j}
    """
    def __init__(self, input_size, hidden_size, normalize, num_layers=1,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(BoundaryPointerLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.lstm = rnn_type(2 * input_size, hidden_size)
        self.answer_attn = PerceptronSeqAttn(hidden_size, normalize)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, batch_size, self.hidden_size).cuda(async=True)),
                Variable(torch.zeros(1, batch_size, self.hidden_size).cuda(async=True)))

    def forward(self, h_hiddens, h_hiddens_mask):
        """
        Args:
            h_hiddens: batch * p_len * 2hdim
            h_hiddens_mask: batch * len (1 for padding, 0 for true)
        Output:
            answer_start_score: p_len
            answer_end_score: p_len
        """
        batch_size, p_len, _ = h_hiddens.size()
        if self.dropout_rate > 0:
            inputs = F.dropout(h_hiddens, p=self.dropout_rate, training=self.training)
        else:
            inputs = h_hiddens

        (ha, ca) = self.init_hidden(batch_size)
        # answer start
        answer_start_score = self.answer_attn(ha.transpose(0, 1), h_hiddens, h_hiddens_mask)  # batch * p_len
        weighted_p = torch.bmm(answer_start_score.unsqueeze(1), h_hiddens)  # batch * 1 * 2hdim
        weighted_p = weighted_p.transpose(0, 1)
        _, (ha, ca) = self.lstm(weighted_p, (ha, ca))
        # answer end
        answer_end_score = self.answer_attn(ha.transpose(0, 1), h_hiddens, h_hiddens_mask)  # batch * p_len
        # if training use log softmax
        if self.training:
            answer_start_score = torch.log(answer_start_score.add(1e-8))
            answer_end_score = torch.log(answer_end_score.add(1e-8))
        return answer_start_score, answer_end_score


class PerceptronSeqAttn(nn.Module):
    """
    Perceptron attention between a series of sequence (matrix) (H^r) and one vector (h^a_k):

    F_k = tanh(VH^r + (W^a h_{k-1}^a + b^a) \otimes e_{(P+1)})
    beta_i = softmax(w^T * G_i + b\otimes e_Q)

    """

    def __init__(self, h_size, normalize=True):
        super(PerceptronSeqAttn, self).__init__()
        self.linear_a = nn.Linear(h_size, h_size)
        self.linear_v = nn.Linear(2 * h_size, h_size, bias=False)
        self.linear_beta = nn.Linear(h_size, 1)
        self.normalize = normalize

    def forward(self, ha, hr, hr_mask):
        """
        Args:
            ha: batch * 1 * hdim
            hr: batch * p_len * 2hdim
            hr_mask: batch * p_len (1 for padding, 0 for true)
        Output:
            beta = batch * p_len
        """
        batch, seq_len, _ = hr.size()
        attn = self.linear_a(ha)  # batch * 1 * hdim
        expn = attn.expand([attn.size(0), seq_len, attn.size(2)])  # batch * seq_len * hdim
        f = F.tanh(self.linear_v(hr) + expn)  # batch * seq_len * hdim
        score = self.linear_beta(f).squeeze(2)  # batch * seq_len
        score.data.masked_fill_(hr_mask.data, -float('inf'))

        padding = torch.sum(hr_mask.data.eq(1).long().sum(1).squeeze())
        if self.normalize:
            beta = F.softmax(score)  # for next step's attn, we dont log here.
        else:
            beta = score.exp()
        return beta


class MatchLSTMLayer(nn.Module):
    """
    Bi-directional LSTM:

    \vec{\mathbf{h}}_i^r = LSTM(\vec{\mathbf{z}}_i, \vec{\mathbf{h}}_{i-1}^r)
    \vec{\mathbf z}_i = \begin{bmatrix} \mathbf{h}_i^p \\ \mathbf{H}^q\vec{\alpha}_i^T\end{bmatrix}
    alpha is computed by MatchAttn
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(MatchLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.flstm = rnn_type(2 * input_size, hidden_size)
        self.blstm = rnn_type(2 * input_size, hidden_size)
        self.match_attn = MatchAttn(hidden_size)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, batch_size, self.hidden_size).cuda(async=True)),
                Variable(torch.zeros(1, batch_size, self.hidden_size).cuda(async=True)))

    def forward(self, q_hiddens, q_hiddens_mask, p_hiddens, p_hiddens_mask):
        """
        Args:
            q_hiddens: batch * q_len * 2hdim
            p_hiddens: batch * p_len * 2hdim
            q_hiddens_mask: batch * len (1 for padding, 0 for true)
            p_hiddens_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        _, p_len, _ = p_hiddens.size()
        batch_size, q_len, _ = q_hiddens.size()

        inv_idx = Variable(torch.arange(p_hiddens.size(1) - 1, -1, -1).long())
        rp_hiddens = p_hiddens.clone()
        # reverse concrete hiddens in rp_hiddens
        lengths = p_hiddens_mask.data.eq(0).long().sum(1).squeeze()  # batch

        for i in range(batch_size):
            if lengths[i] < p_len:  # if lengths[i] == p_len, do nothing
                indices = Variable(torch.cat((torch.arange(lengths[i] - 1, -1, -1),
                                              torch.arange(lengths[i], p_len))).long().cuda())
                rp_hiddens[i, :, :] = rp_hiddens[i, :, :].index_select(0, indices)

        (hr, cr) = self.init_hidden(batch_size)
        (rhr, rcr) = self.init_hidden(batch_size)
        h_hiddens = [hr]
        rh_hiddens = [rhr]

        if self.dropout_rate > 0:
            inputs = F.dropout(p_hiddens, p=self.dropout_rate, training=self.training)
            rinputs = F.dropout(rp_hiddens, p=self.dropout_rate, training=self.training)
        else:
            inputs = p_hiddens
            rinputs = rp_hiddens

        for i in range(p_len):
            alpha = self.match_attn(inputs[:, i, :].unsqueeze(1), hr.transpose(0, 1), q_hiddens,
                                    q_hiddens_mask)  # batch * q_len
            ralpha = self.match_attn(rinputs[:, i, :].unsqueeze(1), rhr.transpose(0, 1), q_hiddens,
                                     q_hiddens_mask)

            weighted_q = torch.bmm(alpha.unsqueeze(1), q_hiddens)  # batch * 1 * hdim
            rweighted_q = torch.bmm(ralpha.unsqueeze(1), q_hiddens)  # batch * 1 * hdim

            flstm_input = torch.cat((inputs[:, i, :].unsqueeze(1), weighted_q), dim=2)  # batch * 1 * 2hdim
            flstm_input = torch.transpose(flstm_input, 0, 1)  # 1 * batch * 2hdim
            _, (hr, cr) = self.flstm(flstm_input, (hr, cr))
            h_hiddens.append(hr)  # 1 * batch * hdim

            blstm_input = torch.cat((rinputs[:, i, :].unsqueeze(1), rweighted_q), dim=2)  # batch * 1 * 2hdim
            blstm_input = torch.transpose(blstm_input, 0, 1)  # 1 * batch * 2hdim
            _, (rhr, rcr) = self.blstm(blstm_input, (rhr, rcr))
            rh_hiddens.append(rhr)  # 1 * batch * hdim

        h_hiddens = torch.cat(h_hiddens[1:], 0)  # p_len * batch * hdim
        h_hiddens = torch.transpose(h_hiddens, 0, 1)  # batch * p_len * hdim

        rh_hiddens = torch.cat(rh_hiddens[1:], 0)  # p_len * batch * hdim
        rh_hiddens = torch.transpose(rh_hiddens, 0, 1)  # batch * p_len * hdim

        # reverse concrete hiddens in rp_hiddens
        for i in range(batch_size):
            if lengths[i] < p_len:
                indices = Variable(torch.cat((torch.arange(lengths[i] - 1, -1, -1),
                                              torch.arange(lengths[i], p_len))).long().cuda())
                rh_hiddens[i, :, :] = rh_hiddens[i, :, :].index_select(0, indices)

        h_hiddens = torch.cat((h_hiddens, rh_hiddens), dim=2)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            h_hiddens = F.dropout(h_hiddens,
                                  p=self.dropout_rate,
                                  training=self.training)

        return h_hiddens


class MatchAttn(nn.Module):
    """
    Match attention between question (H^q) and one passage word(h^p_i):

    G_i = tanh(W^q*H^q + (W^p*h_i^p + W^r*h_{i-1}^r + b^p)\otimes e_Q)
    alpha_i = softmax(w^T * G_i + b\otimes e_Q)

    """

    def __init__(self, h_size):
        super(MatchAttn, self).__init__()
        # combine W^p and W^r
        self.linear_pr = nn.Linear(2 * h_size, h_size)
        self.linear_q = nn.Linear(h_size, h_size, bias=False)
        self.linear_g = nn.Linear(h_size, 1)

    def forward(self, hp, hr, hq, hq_mask):
        """
        Args:
            hp: batch * 1 * hdim
            hr: batch * 1 * hdim
            hq: batch * q_len * hdim
            hq_mask: batch * q_len (1 for padding, 0 for true)
        Output:
            alpha = batch * q_len
        """
        batch, seq_len, _ = hq.size()
        attn = self.linear_pr(torch.cat([hp, hr], 2))  # batch * 1 * hdim
        expn = attn.expand([attn.size(0), seq_len, attn.size(2)])  # batch * seq_len * hdim
        g = F.tanh(self.linear_q(hq) + expn)  # batch * seq_len * hdim
        score = self.linear_g(g).squeeze(2)  # batch * seq_len
        score.data.masked_fill_(hq_mask.data, -float('inf'))
        alpha = F.softmax(score)
        return alpha

class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # batch * len1 * len2

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # batch * 1 * len2 ==> batch * len1 * len2
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))  # batch * len2
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))  # batch * len1 * len2

        # Take weighted average
        matched_seq = alpha.bmm(y)  # batch * len1 * hdim
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y  # Wy  batch * x_size
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)  # Wy.unsqueeze(2)  batch * x_size * 1
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
