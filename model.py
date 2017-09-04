# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class CNNLayers(nn.Module):
    """
    CNN
    """

    def __init__(self):
        super(CNNLayers, self).__init__()
        # 100X32
        self.conv1 = nn.Conv2d(3,   64,  3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        return x


class Encoder(nn.Module):

    """LSTM Encoder"""

    def __init__(self, input_size, num_layers, is_bidirectional, hidden_size):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.directions = 2 if is_bidirectional else 1
        self.is_bidirectional = is_bidirectional
        self.hidden_size = hidden_size

        # hidden_size <- CNN output sequence size
        # Batch First ??
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=is_bidirectional
        )

    def init_state(self):
        h0_encoder = Variable(
            torch.zeros(self.directions * self.num_layers, 1, self.hidden_size),
            required_grad=False
        )
        c0_encoder = Variable(
            torch.zeros(self.directions * self.num_layers, 1, self.hidden_size),
            required_grad=False
        )

        return h0_encoder, c0_encoder

    def get_output_state(self, src_h_t):
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
        return h_t

    def forward(self, input, hidden):

        #s_len, n_batch, n_feats = input.size()
        outputs, hidden_t = self.lstm(
            input,
            hidden
        )

        return outputs, hidden_t


class Attention(nn.Module):

    def __init__(self, hidden_size, max_length):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len))
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = hidden.dot(energy)
        return energy


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, max_length):
        super(Decoder, self).__init__()
        self.attend = Attention(hidden_size,max_length)

        self.hidden_size =hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )

        self.gru = nn.Linear(hidden_size*2, output_size)

    def forward(self, input, last_context, hidden, encoder_outputs):

        input_emb = self.embedding(input).view(1, 1, -1)

        rnn_input = torch.cat((input_emb, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, hidden)

        attn_weights = self.attend(rnn_output.squeeze(0), encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)

        context = context.squeeze(1)

        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        return output, context, hidden, attn_weights














    # class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, input_size, hidden_size):
#         super(BidirectionalLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True,
#         )
#
#     def get_state(self, input):
#         batch_size = input.input_size(0)
#         h = Variable(
#             torch.zero(2, 1, self.hidden_size)
#         )
#
#         c = Variable(
#             torch.zero(2, 1, self.hidden_size)
#         )
#         return h, c
#
#     def forward(self, x):
#         h, c = self.get_state(x)
#         output, hn, cn = self.lstm(x, (h, c))
#         return output
#
#
# class Attention(nn.Module):
#
#     def __init__(self, dim):
#         super(Attention, self).__init__()
#         self.linear_out = nn.Linear(dim*2, dim)
#         self.mask = None
#
#     def set_mask(self, mask):
#         """
#         Sets indices to be masked
#         Args:
#             mask (torch.Tensor): tensor containing indices to be masked
#         """
#         self.mask = mask
#
#     def forward(self, output, context):
#         batch_size = output.size(0)
#         hidden_size = output.size(2)
#         input_size = context.size(1)
#         # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
#         attn = torch.bmm(output, context.transpose(1, 2))
#         if self.mask is not None:
#             attn.data.masked_fill_(self.mask, -float('inf'))
#         attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)
#
#         # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
#         mix = torch.bmm(attn, context)
#
#         # concat -> (batch, out_len, 2*dim)
#         combined = torch.cat((mix, output), dim=2)
#         # output -> (batch, out_len, dim)
#         output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
#
#         return output, attn
#
#
#

#
#
# class LSTMEncoder(nn.Module):
#
#     def __init__(self, is_bidirectional, input_size, encoder_hidden_size=256, num_layers=2):
#         super(LSTMEncoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = encoder_hidden_size
#         self.is_bidirectional = is_bidirectional
#
#         self.encoder = nn.LSTM(
#             input_size=input_size,
#             hidden_size=encoder_hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=is_bidirectional
#         )
#
#         self.num_layers = num_layers
#         self.directions = 2 if is_bidirectional else 1
#
#     def init_state(self):
#         h0_encoder = Variable(
#             torch.zeros(self.directions * self.num_layers, 1, self.hidden_size),
#             required_grad=False
#         )
#         c0_encoder = Variable(
#             torch.zeros(self.directions * self.num_layers, 1, self.hidden_size),
#             required_grad=False
#         )
#
#         return h0_encoder, c0_encoder
#
#     def get_output_state(self, src_h_t):
#         if self.bidirectional:
#             h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
#         else:
#             h_t = src_h_t[-1]
#         return h_t
#
#     def forward(self, input_src, h, c):
#
#         # h0, c0 = self.get_state(input_src)
#
#         src_out, (src_h_t, src_c_t) = self.encoder(
#             input_src,
#             (h, c)
#         )
#
#         # if self.bidirectional:
#         #     h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
#         #     c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
#         # else:
#         #     h_t = src_h_t[-1]
#         #     c_t = src_c_t[-1]
#
#         return src_out, src_h_t, src_c_t
#
#
# class SRNEncoder(nn.Module):
#
#     def __init__(self, is_bidirectional, input_size, encoder_hidden_size=256, num_layers=2):
#         super(SRNEncoder, self).__init__()
#
#         self.cnn = CNNLayers()
#         self.lstm_encoder = LSTMEncoder(is_bidirectional=is_bidirectional,
#                                         input_size=input_size,
#                                         encoder_hidden_size=encoder_hidden_size,
#                                         num_layers=num_layers)
#
#     def forward(self, image, h, c):
#
#         x = self.cnn(image)
#         out, h_t, c_t = self.lstm_encoder(x, h, c)
#
#         return out, h_t, c_t
#
#
#
#
#
# class GRUDecoder(nn.Module):
#     def __init__(self, hidden_size, max_length, output_size, n_layers=1, dropout_p=0.1):
#         super(GRUDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_output, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)))
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         for i in range(self.n_layers):
#             output = F.relu(output)
#             output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]))
#         return output, hidden, attn_weights
#
#     def init_hidden(self):
#         result = Variable(torch.zeros(1, 1, self.hidden_size))
#         return result
#
#
#
#
#
# # class GruAttendDecoder(nn.Module):
# #
# #     def __init__(self, hidden_size, output_size, max_length):
# #         super(GruAttendDecoder, self).__init__()
# #         self.hidden_size = hidden_size
# #         self.output_size = output_size
# #         self.max_length = max_length
# #
# #         self.gru = nn.GRU(
# #             input_size=hidden_size,
# #             hidden_size=hidden_size
# #         )
# #
# #         self.out = nn.Linear(
# #             hidden_size,
# #             output_size
# #         )
# #
# #     def forward(self, input, hidden, encoder_output, encoder_outputs):
# #
#
# class AttentionDecoderGRU(nn.Module):
#     def __init__(self, output_size, max_length, hidden_size=16, n_layers=1, dropout_p=0.1):
#         super(AttentionDecoderGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         # Initialize nn modules
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_output, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         # Global Attention
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)))
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         # GRU layer
#         for i in range(self.n_layers):
#             output = F.relu(output)
#             output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]))
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         result = Variable(torch.zeros(1, 1, self.hidden_size))
#         return result
#
#
# class SoftDotAttention(nn.Module):
#     """Soft Dot Attention.
#
#     Ref: http://www.aclweb.org/anthology/D15-1166
#     Adapted from PyTorch OPEN NMT.
#     """
#
#     def __init__(self, dim):
#         """Initialize layer."""
#         super(SoftDotAttention, self).__init__()
#         self.linear_in = nn.Linear(dim, dim, bias=False)
#         self.sm = nn.Softmax()
#         self.linear_out = nn.Linear(dim * 2, dim, bias=False)
#         self.tanh = nn.Tanh()
#         self.mask = None
#
#     def forward(self, input, context):
#         """Propogate input through the network.
#
#         input: batch x dim
#         context: batch x sourceL x dim
#         """
#         target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
#
#         # Get attention
#         attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
#         attn = self.sm(attn)
#         attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
#
#         weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
#         h_tilde = torch.cat((weighted_context, input), 1)
#
#         h_tilde = self.tanh(self.linear_out(h_tilde))
#
#         return h_tilde, attn
#
#
# class GRUAttentionDot(nn.Module):
#     r"""A gate recurrent unit cell with attention."""
#
#     def __init__(self, input_size, hidden_size, batch_first=True):
#         """Initialize params."""
#         super(GRUAttentionDot, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = 1
#         self.batch_first = batch_first
#
#         self.input_weights = nn.Linear(input_size, 3 * hidden_size)
#         self.hidden_weights = nn.Linear(hidden_size, 3 * hidden_size)
#
#         self.attention_layer = SoftDotAttention(hidden_size)
#
#     def forward(self, input, hidden, ctx, ctx_mask=None):
#         """Propogate input through the network."""
#         def recurrence(input, hidden):
#             """Recurrence helper."""
#             hx = hidden  # n_b x hidden_dim
#             gates = self.input_weights(input) + \
#                 self.hidden_weights(hx)
#             reset_gate, update_gate, alternative_gate = gates.chunk(3, 1)
#
#
#
#
#             cy = (forgetgate * cx) + (ingate * cellgate)
#             hy = outgate * F.tanh(cy)  # n_b x hidden_dim
#             h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))
#
#             return h_tilde, cy
#
#         if self.batch_first:
#             input = input.transpose(0, 1)
#
#         output = []
#         steps = range(input.size(0))
#         for i in steps:
#             hidden = recurrence(input[i], hidden)
#             output.append(isinstance(hidden, tuple) and hidden[0] or hidden)
#
#         output = torch.cat(output, 0).view(input.size(0), *output[0].size())
#
#         if self.batch_first:
#             output = output.transpose(0, 1)
#
#         return output, hidden
#
#
#
#
# class Seq2SeqAttention(nn.Module):
#
#     def __init__(self,
#                  is_bidirectional,
#                  input_size,
#                  trg_hidden_dim,
#                  output_size,
#                  trg_input_size,
#                  max_length,
#                  encoder_hidden_size=256,
#                  decoder_hidden_size=256,
#                  num_layers=2,
#                  ):
#         super(Seq2SeqAttention, self).__init__()
#         self.is_bidirectional = is_bidirectional
#         self.input_size = input_size
#         self.hidden_size = encoder_hidden_size
#
#         self.encoder = nn.LSTM(
#             input_size=input_size,
#             hidden_size=encoder_hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=is_bidirectional
#         )
#
#         self.num_layers = num_layers
#         self.directions = 2 if is_bidirectional else 1
#
#         self.attend = nn.Linear(
#             encoder_hidden_size * self.directions,
#             trg_hidden_dim
#         )
#
#         self.decoder = nn.GRU(
#             input_size=trg_input_size,
#             batch_first=True,
#             hidden_size=decoder_hidden_size
#         )
#
#     def get_state(self, input):
#         batch_size = input.size(0)
#         h0_encoder = Variable(
#             torch.zeros(self.directions * self.num_layers, batch_size, self.hidden_size),
#             required_grad=False
#         )
#         c0_encoder = Variable(
#             torch.zeros(self.directions * self.num_layers, batch_size, self.hidden_size),
#             required_grad=False
#         )
#
#         return h0_encoder, c0_encoder
#
#     def forward(self, input_src, input_trg):
#
#         self.h0_lstm, self.c0_lstm = self.get_state(input_src)
#
#         src_out, (src_h_t, src_c_t) = self.encoder(
#             input_src,
#             (self.h0_lstm, self.c0_lstm)
#         )
#
#         if self.bidirectional:
#             h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
#             c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
#         else:
#             h_t = src_h_t[-1]
#             c_t = src_c_t[-1]
#
#         decoder_init_state = self.attend(h_t)
#         decoder_init_state = F.tanh(decoder_init_state)




