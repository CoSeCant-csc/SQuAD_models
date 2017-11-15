#!/usr/bin/env python3
"""Implementation of the Bi-directional attention flow reader."""

import torch
import torch.nn as nn
from . import layers
from ..module import CnnEncoder, Highway, TimeDistributed, MatrixAttention, util
from ..module.similarity_functions import LinearSimilarity


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class BidafDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(BidafDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        # Char embeddings (+1 for padding)
        self.char_embedding = nn.Embedding(args.character_vocab_size,
                                           args.char_embedding_dim,
                                           padding_idx=0)

        char_cnn_ngram_filter_sizes = [int(size) for size in args.char_cnn_ngram_filter_sizes]
        # character encoder
        self.char_cnn = CnnEncoder(args.char_embedding_dim,
                                   args.char_cnn_num_filters,
                                   char_cnn_ngram_filter_sizes)

        self.args.char_cnn_embedding_dim = len(args.char_cnn_ngram_filter_sizes) * args.char_cnn_num_filters

        self.highway_layer = TimeDistributed(Highway(args.embedding_dim + args.char_cnn_embedding_dim,
                                                     args.highway_layers))
        # Input size to RNN: word emb + cnn emb + manual features
        highway_output_size = args.embedding_dim + args.char_cnn_embedding_dim
        doc_input_size = highway_output_size + args.num_features

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,  # for highway layer
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=highway_output_size,  # for highway layer
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders (bi-directional)
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        self.matrix_attention = MatrixAttention(LinearSimilarity(doc_hidden_size,
                                                                 question_hidden_size, 'x,y,x*y'))

        self.modeling_layer = layers.StackedBRNN(
            input_size=doc_hidden_size * 4,
            hidden_size=args.hidden_size,
            num_layers=args.modeling_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        modeling_hidden_size = 2 * args.hidden_size

        self.span_end_encoder = layers.StackedBRNN(
            input_size=doc_hidden_size * 4 + modeling_hidden_size * 3,
            hidden_size=args.hidden_size,
            num_layers=args.span_end_encode_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        span_end_encoding_dim = 2 * args.hidden_size

        span_start_input_dim = doc_hidden_size * 4 + modeling_hidden_size
        self.span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_input_dim = doc_hidden_size * 4 + span_end_encoding_dim
        self.span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

    def forward(self, x1, x1_mask, x1_char, x1_char_mask, x1_f, x2, x2_mask, x2_char, x2_char_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x1_char = document chars indices       [batch * len_d * len_w]
        x1_char_mask = document chars padding mask       [batch * len_d * len_w]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        x2_char = question chars indices       [batch * len_q * len_w]
        x2_char_mask = question chars padding mask       [batch * len_q * len_w]
        """
        # Embed both document and question
        x1_word_emb = self.embedding(x1)  # [batch, len_d, embedding_dim]
        x2_word_emb = self.embedding(x2)  # [batch, len_q, embedding_dim]

        # Get char embeddings: [batch*len_d, len_w, char_embedding_dim]
        x1_char_emb = self.char_embedding(x1_char.view(-1, x1_char.size(-1)))
        # shape: [batch*len_d, num_conv_layers * num_filters = cnn_emb_dim]
        x1_char_emb = self.char_cnn(x1_char_emb)
        # shape: [batch, len_d, cnn_emb_dim]
        x1_char_emb = x1_char_emb.view(-1, x1_char.size(1), x1_char_emb.size(-1))

        # Get char embeddings: [batch*len_q, len_w, char_embedding_dim]
        x2_char_emb = self.char_embedding(x2_char.view(-1, x2_char.size(-1)))
        # shape: [batch*len_q, num_conv_layers * num_filters = cnn_emb_dim]
        x2_char_emb = self.char_cnn(x2_char_emb)
        # shape: [batch, len_q, cnn_emb_dim]
        x2_char_emb = x2_char_emb.view(-1, x2_char.size(1), x2_char_emb.size(-1))

        x1_emb = torch.cat([x1_word_emb, x1_char_emb], dim=-1)
        x2_emb = torch.cat([x2_word_emb, x2_char_emb], dim=-1)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)
        x1_emb = self.highway_layer(x1_emb)
        x2_emb = self.highway_layer(x2_emb)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN shape: [batch, len_d, 2*hidden_size]
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)
        # Encode question with RNN + merge hiddens shape: [batch, len_q, 2*hidden_size]
        question_hiddens = self.question_rnn(x2_emb, x2_mask)

        # shape: [batch, len_d, len_q]
        doc_question_similarity = self.matrix_attention(doc_hiddens, question_hiddens)
        # shape: [batch, len_d, len_q]
        doc_question_attention = util.last_dim_softmax(doc_question_similarity, x2_mask)  # this mask is reverse of
        # the allennlp one
        # shape: [batch, len_d, 2*hidden_size]
        doc_question_vectors = util.weighted_sum(question_hiddens, doc_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        # shape: [batch, len_d, len_q]
        masked_similarity = util.replace_masked_values(doc_question_similarity,
                                                       x2_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch, len_d)
        question_doc_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch, len_d)
        question_doc_attention = util.masked_softmax(question_doc_similarity, x1_mask)
        # Shape: (batch, 2*hidden_size)
        question_passage_vector = util.weighted_sum(doc_hiddens, question_doc_attention)
        # Shape: (batch, len_d, 2*hidden_size)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(doc_hiddens.size())

        # Shape: (batch, len_d, 2*hidden_size * 4)
        final_merged_doc = torch.cat([doc_hiddens,
                                      doc_question_vectors,
                                      doc_hiddens * doc_question_vectors,
                                      doc_hiddens * tiled_question_passage_vector],
                                     dim=-1)

        # Shape: (batch, len_d, 2*hidden_size)
        modeled_doc = self.modeling_layer(final_merged_doc, x1_mask)
        # Shape: (batch_size, len_d, 2*hidden_size * 4 + 2*hidden_size))
        span_start_input = torch.cat([final_merged_doc, modeled_doc], dim=-1)
        # Shape: (batch_size, len_d)
        span_start_logits = self.span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, len_d)
        span_start_probs = util.masked_softmax(span_start_logits, x1_mask)

        # Shape: (batch, 2*hidden_size)
        span_start_representation = util.weighted_sum(modeled_doc, span_start_probs)
        # Shape: (batch, len_d, 2*hidden_size)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(modeled_doc.size())

        # Shape: (batch, len_d, 2*hidden_size * 4 + 2*hidden_size * 3)
        span_end_representation = torch.cat([final_merged_doc,
                                             modeled_doc,
                                             tiled_start_representation,
                                             modeled_doc * tiled_start_representation],
                                            dim=-1)
        # Shape: (batch_size, len_d, 2*hidden_size)
        encoded_span_end = self.span_end_encoder(span_end_representation, x1_mask)
        # Shape: (batch_size, len_d, 2*hidden_size * 4 + 2*hidden_size)
        span_end_input = torch.cat([final_merged_doc, encoded_span_end], dim=-1)
        # Shape: (batch_size, len_d)
        span_end_logits = self.span_end_predictor(span_end_input).squeeze(-1)
        # Shape: (batch_size, len_d)
        span_end_probs = util.masked_softmax(span_end_logits, x1_mask)

        if self.training:
            span_start_probs = util.masked_log_softmax(span_start_logits, x1_mask)
            span_end_probs = util.masked_log_softmax(span_end_logits, x1_mask)

        return span_start_probs, span_end_probs
