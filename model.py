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
        )

    def get_state(self, input):
        batch_size = self.input_size(0)
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




class SRN(nn.Module):

    def __init__(self):
        super(SRN, self).__init__()
        # 100X32
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)

        self.blstm1 = nn.LSTM(512, 256, num_layers=1, bidirectional=True)
        self.blstm2 = nn.LSTM(256, 256, num_layers=1, bidirectional=True)

        self.attend = nn.Linear()



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




