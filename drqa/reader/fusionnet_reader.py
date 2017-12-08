#!/usr/bin/env python3
"""Implementation of the FusionNet reader."""

import torch
import torch.nn as nn
from . import layers
from cove import MTLSTM
from ..module import MatrixAttention, util
from ..module.similarity_functions import SymmetricBilinearSimilarity


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class FusionNetReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(FusionNetReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        if args.use_cove and args.embedding_dim == 300:
            # init cove_encoder without additional embeddings
            self.cove_encoder = MTLSTM()  # 300

        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + cove emb + manual features + question emb
        doc_input_size = args.embedding_dim + args.num_features
        question_input_size = args.embedding_dim
        if args.use_cove:
            doc_input_size += 2 * args.cove_embedding_dim
            question_input_size += 2 * args.cove_embedding_dim
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # Reading component (low-level layer)
        self.reading_low_level_doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            padding=args.rnn_padding
        )

        self.reading_low_level_question_rnn = layers.StackedBRNN(
            input_size=question_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            padding=args.rnn_padding
        )

        # Reading component (high-level layer)
        self.reading_high_level_doc_rnn = layers.StackedBRNN(
            input_size=args.hidden_size * 2,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            padding=args.rnn_padding
        )

        self.reading_high_level_question_rnn = layers.StackedBRNN(
            input_size=args.hidden_size * 2,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            padding=args.rnn_padding
        )

        # Question understanding component
        # input: [low_level_question, high_level_question]
        self.understanding_question_rnn = layers.StackedBRNN(
            input_size=args.hidden_size * 4,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            padding=args.rnn_padding
        )

        # [word_embedding, cove_embedding, low_level_doc_hidden, high_level_doc_hidden]
        history_of_word_size = args.embedding_dim + 2 * args.cove_embedding_dim + 4 * args.hidden_size

        self.low_level_matrix_attention = MatrixAttention(SymmetricBilinearSimilarity(history_of_word_size,
                                                                                      args.attention_size,
                                                                                      torch.nn.ReLU))
        self.high_level_matrix_attention = MatrixAttention(SymmetricBilinearSimilarity(history_of_word_size,
                                                                                       args.attention_size,
                                                                                       torch.nn.ReLU))
        self.understanding_matrix_attention = MatrixAttention(SymmetricBilinearSimilarity(history_of_word_size,
                                                                                         args.attention_size,
                                                                                         torch.nn.ReLU))

        # Multi-level rnn
        # input: [low_level_doc, high_level_doc, low_level_fusion_doc, high_level_fusion_doc,
        # understanding_level_question_fusion_doc]
        self.multi_level_rnn = layers.StackedBRNN(
            input_size=args.hidden_size * 2 * 5,
            hidden_size=args.hidden_size,
            num_layers=1,
            padding=args.rnn_padding
        )

        # [word_embedding, cove_embedding, low_level_doc_hidden, high_level_doc_hidden, low_level_doc_question_vector,
        # high_level_doc_question_vector, understanding_doc_question_vector, fa_multi_level_doc_hidden]
        history_of_doc_word_size = history_of_word_size + 4 * 2 * args.hidden_size

        self.self_boosted_matrix_attention = MatrixAttention(SymmetricBilinearSimilarity(history_of_doc_word_size,
                                                                                         args.attention_size,
                                                                                         torch.nn.ReLU))
        # Fully-Aware Self-Boosted fusion rnn
        # input: [fully_aware_encoded_doc(hidden state from last layer) ,self_boosted_fusion_doc]
        self.understanding_doc_rnn = layers.StackedBRNN(
            input_size=args.hidden_size * 2 * 2,
            hidden_size=args.hidden_size,
            num_layers=1,
            padding=args.rnn_padding
        )

        # Output sizes of rnn
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        self.question_self_attn = layers.LinearSeqAttn(question_hidden_size)

        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)

        self.start_gru = nn.GRU(doc_hidden_size, args.hidden_size, 1)

        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_word_emb = self.embedding(x1)  # [batch, len_d, embedding_dim]
        x2_word_emb = self.embedding(x2)  # [batch, len_q, embedding_dim]

        x1_lengths = x1_mask.data.eq(0).long().sum(1).squeeze()  # batch
        x2_lengths = x2_mask.data.eq(0).long().sum(1).squeeze()  # batch

        x1_cove_emb = self.cove_encoder(x1_word_emb, x1_lengths)
        x2_cove_emb = self.cove_encoder(x2_word_emb, x2_lengths)

        x1_emb = torch.cat([x1_word_emb, x1_cove_emb], dim=-1)
        x2_emb = torch.cat([x2_word_emb, x2_cove_emb], dim=-1)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)
        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_word_emb, x1_word_emb, x2_mask)  # batch * len_d
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN shape: [batch, len_d, 2*hidden_size]
        low_level_doc_hiddens = self.reading_low_level_doc_rnn(torch.cat(drnn_input, 2), x1_mask)
        low_level_question_hiddens = self.reading_low_level_question_rnn(x2_emb, x2_mask)

        # Encode question with RNN shape: [batch, len_q, 2*hidden_size]
        high_level_doc_hiddens = self.reading_high_level_doc_rnn(low_level_doc_hiddens, x1_mask)
        high_level_question_hiddens = self.reading_high_level_question_rnn(low_level_question_hiddens, x2_mask)

        # Encode low_level_question_hiddens and high_level_question_hiddens shape:[batch, len_q, 2*hidden_size]
        understanding_question_hiddens = self.understanding_question_rnn(torch.cat([low_level_question_hiddens,
                                                                                    high_level_question_hiddens], 2),
                                                                         x2_mask)

        # history of word shape:[batch, len_d, history_of_word_size]
        history_of_doc_word = torch.cat([x1_emb, x1_cove_emb, low_level_doc_hiddens, high_level_doc_hiddens]
                                        , dim=2)
        # history of word shape:[batch, len_q, history_of_word_size]
        history_of_question_word = torch.cat([x2_emb, x2_cove_emb, low_level_question_hiddens,
                                              low_level_question_hiddens], dim=2)

        # fully-aware multi-level attention
        low_level_similarity = self.low_level_matrix_attention(history_of_doc_word, history_of_question_word)
        high_level_similarity = self.high_level_matrix_attention(history_of_doc_word, history_of_question_word)
        understanding_similarity = self.understanding_matrix_attention(history_of_doc_word, history_of_question_word)

        # shape: [batch, len_d, len_q]
        low_level_norm_sim = util.last_dim_softmax(low_level_similarity, x2_mask)
        high_level_norm_sim = util.last_dim_softmax(high_level_similarity, x2_mask)
        understanding_norm_sim = util.last_dim_softmax(understanding_similarity, x2_mask)

        # shape: [batch, len_d, 2*hidden_size]
        low_level_doc_question_vectors = util.weighted_sum(low_level_question_hiddens, low_level_norm_sim)
        high_level_doc_question_vectors = util.weighted_sum(high_level_question_hiddens, high_level_norm_sim)
        understanding_doc_question_vectors = util.weighted_sum(understanding_question_hiddens, understanding_norm_sim)

        # Encode multi-level hiddens and vectors
        fa_multi_level_doc_hiddens = self.multi_level_rnn(torch.cat([low_level_doc_hiddens, high_level_doc_hiddens,
                                                                     low_level_doc_question_vectors,
                                                                     high_level_doc_question_vectors,
                                                                     understanding_doc_question_vectors]), dim=2)
        #
        history_of_doc_word = torch.cat([x1_emb, x1_cove_emb, low_level_doc_hiddens, high_level_doc_hiddens,
                                         low_level_doc_question_vectors, high_level_doc_question_vectors,
                                         understanding_doc_question_vectors, fa_multi_level_doc_hiddens], dim=2)

        # shape: [batch, len_d, len_d]
        self_boosted_similarity = self.self_boosted_matrix_attention(history_of_doc_word, history_of_doc_word)

        # shape: [batch, len_d, len_d]
        self_boosted_norm_sim = util.last_dim_softmax(self_boosted_similarity, x1_mask)

        # shape: [batch, len_d, 2*hidden_size]
        self_boosted_vectors = util.weighted_sum(fa_multi_level_doc_hiddens, self_boosted_norm_sim)

        # Encode vectors and hiddens
        # shape: [batch, len_d, 2*hidden_size]
        understanding_doc_hiddens = self.understanding_doc_rnn(torch.cat([fa_multi_level_doc_hiddens,
                                                                          self_boosted_vectors], dim=2))

        # shape: [batch, len_q]
        q_merge_weights = self.question_self_attn(understanding_question_hiddens, x2_mask)
        # shape: [batch, 2*hidden_size]
        question_hidden = layers.weighted_avg(understanding_question_hiddens, q_merge_weights)

        # Predict start and end positions
        # shape: [batch, len_d]
        start_scores = self.start_attn(understanding_doc_hiddens, question_hidden, x1_mask)
        # shape: [batch, 2*hidden_size]
        gru_input = layers.weighted_avg(understanding_doc_hiddens, start_scores)
        _, memory_hidden = self.start_gru(gru_input, question_hidden)
        end_scores = self.end_attn(understanding_doc_hiddens, memory_hidden, x1_mask)

        return start_scores, end_scores
