# -*- coding: utf-8 -*-

import torch
import torch.optim as optimizer
from model import *
import random
from data import TestDataset
import torch.utils.data as torch_data


EOS = 1


class TrainLog(object):

    def __init__(self):
        pass


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
        self.epoches = config["epoches"]
        self.print_step = config["print_step"]

        self.cnn = CNNLayers()
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        input_size = 512
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

        self.criterion = nn.MSELoss()

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
        decoder_hidden = encoder_hidden#self.encoder.get_output_state(encoder_hidden)
        target = target.squeeze(0)
        target = target.type(torch.FloatTensor)

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
        for param in self.cnn.parameters():
        self.cnn_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0]

    def train(self, data_loader):

        for epoch in range(self.epoches):

            loss = 0.0
            for idx, data in enumerate(data_loader):
                image, label = data
                image = Variable(image.type(torch.FloatTensor))
                label = Variable(label)
                current_loss = self.train_step(image, label)

                loss += current_loss
                if idx % self.print_step == 0:
                    print("EPOCH: %s BATCH: %s LOSS: %s AVG_LOSS: %s" % (epoch, idx, current_loss, loss/(idx+1)))

            print("EPOCH: %s AVG_LOSS: %s" % (epoch, len(data_loader)))

    def evaluate(self):
        pass


def main():

    hyper_paramenters = {
        "learning_rate": 0.001,
        "batch_size": 1,
        "epoches": 100,
        "print_step": 10,
    }

    train_loader = torch_data.DataLoader(dataset=TestDataset(True), batch_size=1, shuffle=True)

    srn = SRN(config=hyper_paramenters, output_size=10, max_length=10)

    srn.train(train_loader)

if __name__ == '__main__':

    main()