# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optimizer


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def get_state(self, input):
        batch_size = input.input_size(0)
        h = Variable(
            torch.zero(2, batch_size, self.hidden_size)
        )

        c = Variable(
            torch.zero(2, batch_size, self.hidden_size)
        )
        return h, c

    def forward(self, x):
        h, c = self.get_state(x)
        output, hn, cn = self.lstm(x, (h, c))
        return output


class Seq2SeqAttention(nn.Module):

    def __init__(self, is_bidirectional, input_size, trg_hidden_dim, hidden_size=256, num_layers=2):
        super(Seq2SeqAttention, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=is_bidirectional
        )

        self.num_layers = num_layers
        self.directions = 2 if is_bidirectional else 1

        self.attend = nn.Linear(
            hidden_size * self.directions,
            trg_hidden_dim
        )

    def get_state(self, input):
        batch_size = input.size(0)
        h0_encoder = Variable(
            torch.zeros(self.directions * self.num_layers, batch_size, self.hidden_size),
            required_grad=False
        )
        c0_encoder = Variable(
            torch.zeros(self.directions * self.num_layers, batch_size, self.hidden_size),
            required_grad=False
        )

        return h0_encoder, c0_encoder

    def forward(self, input_src, input_trg):

        self.h0_lstm, self.c0_lstm = self.get_state(input_src)

        src_out, (src_h_t, src_c_t) = self.lstm(
            input_src,
            (self.h0_lstm, self.c0_lstm)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        ctx = self.attend(h_t)
        ctx = F.tanh(ctx)

        











class SRN(nn.Module):

    def __init__(self):
        super(SRN, self).__init__()
        # 100X32
        self.conv1 = nn.Conv2d(3,   64,  3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,  128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)



    def initHidden(self):
        pass


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




