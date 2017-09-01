# -*- coding: utf-8 -*-

import torch
import torch.optim as optimizer
from model import *



hyper_paramenters = {
    "learning_rate": 0.001,
    "batch_size": 32
}


class SRN(object):

    def __init__(self, config, input_size, output_size, max_length, encoder_hidden_size=256, decoder_hidden_size=256):

        self.encoder = SRNEncoder(True, input_size, encoder_hidden_size, 2)
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder = GRUDecoder(decoder_hidden_size, max_length, output_size, 1)

        self.learning_rate = config['learning_rate']

        self.teacher_forcing_ratio = 0.5

        self.encoder_optimizer = optimizer.RMSprop(lr=self.learning_rate, params=self.encoder.parameters())
        self.decoder_optimizer = optimizer.RMSprop(lr=self.learning_rate, params=self.decoder.parameters())

    def train_step(self, input):
        encoder_state_c, encoder_state_h = self.encoder.lstm_encoder.init_state()
        input_length = 100







    def train(self):
        pass



    def evaluate(self):
        pass
