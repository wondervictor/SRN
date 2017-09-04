# -*- coding: utf-8 -*-

import torch
import torch.optim as optimizer
from model import *
import random

hyper_paramenters = {
    "learning_rate": 0.001,
    "batch_size": 32
}

EOS = 1

class SRN(object):

    def __init__(
            self,
            config,
            output_size,
            max_length,
            encoder_hidden_size=256,
            decoder_hidden_size=256,
            teacher_forcing_ratio=0.5
    ):
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.cnn = CNNLayers()
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        input_size = 1000
        self.encoder = Encoder(
            input_size=input_size,
            num_layers=2,
            is_bidirectional=True,
            hidden_size=encoder_hidden_size
        )

        self.decoder = Decoder(
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            num_layers=1,
            max_length=max_length
        )
        self.cnn_optimizer = optimizer.RMSprop(params=self.cnn.parameters(), lr=self.learning_rate)
        self.encoder_optimizer = optimizer.RMSprop(params=self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optimizer.RMSprop(params=self.decoder.parameters(), lr=self.learning_rate)

        self.criterion = nn.NLLLoss()

    def train_step(self, image, target):

        self.cnn_optimizer.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        loss = 0

        "TODO: map to sequence"
        input_sequence = self.cnn.forward(image)

        input_length = input_sequence.size()[0]
        target_length = target.size()[0]

        encoder_hidden = self.encoder.init_state()

        encoder_outputs, encoder_hidden = self.encoder.forward(input_sequence, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[0]]))
        decoder_context = Variable(torch.zeros(1, self.decoder_hidden_size))
        decoder_hidden = self.encoder.get_output_state(encoder_hidden[0])

        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        if use_teacher_forcing:

            for i in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder.forward(
                    decoder_input,
                    decoder_context,
                    decoder_hidden,
                    encoder_outputs
                )

                loss += self.criterion(decoder_output[0], target[i])
                decoder_input = target[i]
        else:
            for i in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder.forward(
                    decoder_input,
                    decoder_context,
                    decoder_hidden,
                    encoder_outputs
                )

                loss += self.criterion(decoder_output[0], target[i])
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                if ni == EOS:
                    break


        loss.backward()
        self.cnn_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0]


    def train(self):
        pass



    def evaluate(self):
        pass
