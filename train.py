# -*- coding: utf-8 -*-

import torch
import torch.optim as optimizer
from model import *
import random
from data import create_dataset
import torch.utils.data as torch_data


EOS = 1

from torch.nn import functional

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length))

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss




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
        self.output_size = output_size

        input_size = 512
        self.encoder = Encoder(
            input_size=input_size,
            num_layers=2,
            is_bidirectional=True,
            hidden_size=encoder_hidden_size,
            batch_size=self.batch_size
        )

        self.decoder = Decoder(
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            num_layers=1,
            max_length=max_length,
            batch_size=self.batch_size
        )
        self.cnn_optimizer = optimizer.RMSprop(params=self.cnn.parameters(), lr=self.learning_rate)
        self.encoder_optimizer = optimizer.RMSprop(params=self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optimizer.RMSprop(params=self.decoder.parameters(), lr=self.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, input, target, length):
        input = input[:length]
        target = target.squeeze(0)
        target = target[:length]
        sub_loss = self.criterion(input, target)
        return sub_loss

    def check_params(self, layer):
        for param in layer.parameters():
            print(param.grad)

    def train_step(self, image, target, maxlength, target_lengths):

        input_sequence = self.cnn.forward(image)
        # target = target.squeeze(0)
        target = target.transpose(0, 1)

        input_length = input_sequence.size()[0]
        target_length = maxlength #target.shape[0]

        encoder_hidden = self.encoder.init_state()

        encoder_outputs, encoder_hidden = self.encoder.forward(input_sequence, encoder_hidden)
        decoder_input = Variable(torch.LongTensor([0]*self.batch_size))
        decoder_context = Variable(torch.zeros(1, self.batch_size, self.decoder_hidden_size))

        decoder_hidden = encoder_hidden
        #self.encoder.get_output_state(encoder_hidden)
        # target = target.type(torch.FloatTensor)

        all_decoder_outputs = Variable(torch.zeros(maxlength, self.batch_size, self.output_size))

        for i in range(maxlength):
            all_decoder_outputs[i], decoder_context, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            decoder_input = target[:,:,i]

        loss = Variable(torch.FloatTensor([0.]))

        for i in range(self.batch_size):
            sub_loss = self.calculate_loss(all_decoder_outputs[:, i, :], target[:, i, :], length=target_lengths[i])
            loss += sub_loss

        self.cnn_optimizer.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()
        self.check_params(self.decoder)
        self.encoder_optimizer.step()
        self.cnn_optimizer.step()

        return loss.data[0]

    def train(self, dataset):

        size = len(dataset[0])

        for epoch in range(self.epoches):

            idx = 0
            loss = 0.0

            while idx < size:
                image = np.array(dataset[0][idx: idx + self.batch_size])
                label = np.array(dataset[1][idx: idx + self.batch_size])
                idx += self.batch_size
                label_lengths = [len(x) for x in label]
                maxlen = np.max(label_lengths)
                input_label = [s.tolist() + [0] * (maxlen - len(s)) for s in label]
                batch_in = Variable(torch.LongTensor(input_label))
                batch_in = batch_in.unsqueeze(1)
                image = Variable(torch.FloatTensor(image))
                current_loss = self.train_step(image, batch_in, maxlen, label_lengths)

                loss += current_loss
                if idx % self.print_step == 0:
                    print("EPOCH: %s BATCH: %s LOSS: %s AVG_LOSS: %s" % (epoch, idx, current_loss, loss/(idx+1)))

            print("EPOCH: %s AVG_LOSS: %s" % (epoch, loss/size))

    def evaluate(self):
        pass


def main():

    batch_size = 16

    hyper_paramenters = {
        "learning_rate": 0.001,
        "batch_size": batch_size,
        "epoches": 100,
        "print_step": 10,
    }

    #train_loader = torch_data.DataLoader(dataset=TestDataset(True), batch_size=batch_size, shuffle=True)

    srn = SRN(config=hyper_paramenters, output_size=10, max_length=10)

    dataset = create_dataset(True)
    srn.train(dataset)

if __name__ == '__main__':

    main()