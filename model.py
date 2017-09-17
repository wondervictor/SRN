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
        self.conv1 = nn.Conv2d(1,   64,  3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d((2, 1))
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.bn4 = nn.BatchNorm2d(num_features=512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.bn5(x)
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = self.pool2(x)
        x = F.relu(self.conv7(x)).squeeze(2)
        x = self.bn4(x)
        x = x.transpose(1, 2)
        return x


class Encoder(nn.Module):

    """LSTM Encoder"""

    def __init__(self, input_size, batch_size, num_layers, is_bidirectional, hidden_size):
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
            bidirectional=is_bidirectional,
            batch_first=True
        )
        self.batch_size = batch_size
        self.out = nn.Linear(hidden_size*2, hidden_size)
        self.hidden_out = nn.Linear(hidden_size*2, hidden_size)

    def init_state(self):
        h0_encoder = Variable(
            torch.zeros(self.directions * self.num_layers, self.batch_size, self.hidden_size),
        )
        c0_encoder = Variable(
            torch.zeros(self.directions * self.num_layers, self.batch_size, self.hidden_size),
        )

        return h0_encoder, c0_encoder

    def get_output_state(self, src_h_t):

        if self.is_bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
        return h_t.unsqueeze(0)

    def forward(self, input, hidden):

        outputs, (hidden_t, _) = self.lstm(
            input,
            hidden
        )
        hidden_t = torch.cat((hidden_t[-1], hidden_t[-2]), 1)
        outputs = F.relu(self.out(outputs))
        hidden_t = F.tanh(self.hidden_out(hidden_t))

        return outputs, hidden_t.unsqueeze(0)


class Attention(nn.Module):

    def __init__(self, hidden_size, max_length):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]
        attn_energies = Variable(torch.zeros(batch_size, seq_len))

        for i in range(batch_size):
            atten = Variable(torch.zeros(seq_len))
            for j in range(seq_len):
                atten[j] = self.score(hidden[:, i], encoder_outputs[i, j].unsqueeze(0))
            attn_energies[i] = F.softmax(atten)

        return attn_energies.unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output).squeeze(0)
        energy = hidden.squeeze(0).dot(energy)
        return energy


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, batch_size, num_layers, max_length):
        super(Decoder, self).__init__()
        self.attend = Attention(hidden_size, max_length)
        self.batch_size = batch_size
        self.hidden_size =hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size+1, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size*2, output_size)

    def forward(self, input, last_context, hidden, encoder_outputs):
        input_emb = self.embedding(input)
        input_emb = input_emb.view(1, self.batch_size, self.hidden_size)
        rnn_input = torch.cat([input_emb, last_context], 2)

        rnn_output, hidden = self.gru(rnn_input, hidden)

        attn_weights = self.attend(rnn_output, encoder_outputs)
        print('---')
        attn_weights = attn_weights.squeeze(0).transpose(0, 1)
        context = attn_weights.bmm(encoder_outputs)
        context = context.squeeze(1).unsqueeze(0)
        cat = torch.cat((rnn_output, context), 2)

        out = self.out(cat)
        output = F.softmax(out.squeeze(0))
        return output, context, hidden, attn_weights




