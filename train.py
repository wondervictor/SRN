# -*- coding: utf-8 -*-

import torch
import torch.optim as optimizer
from model import *
import random
from data import create_dataset
import torch.utils.data as torch_data


EOS = 10
SOS = 10
PAD = 11


class TrainLog(object):

    def __init__(self):
        pass

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    # if sequence_length.is_cuda:
    #     seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length)) #.cuda()

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
    log_probs_flat = F.log_softmax(logits_flat)
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

        self.decoder = AttnDecoder(
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            n_layers=1,
            max_length=max_length,
        )

        self.cnn_optimizer = optimizer.Adam(params=self.cnn.parameters(), lr=self.learning_rate)
        self.encoder_optimizer = optimizer.Adam(params=self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optimizer.Adam(params=self.decoder.parameters(), lr=self.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, input, target, length):
        input = input[:length]
        target = target.squeeze(0)
        target = target[:length]
        sub_loss = self.criterion(input, target)
        return sub_loss/8

    def pad_seq(self, seq, max_length):
        seq += [PAD]*(max_length - len(seq))
        return seq

    def check_params(self, layer):
        for param in layer.parameters():
            print(param.grad)

    def train_step(self, image_batches, target_batches, target_lengths):
        max_target_length = max(target_lengths)
        input_sequence = self.cnn.forward(image_batches)
        target = target_batches.transpose(0, 1)
        encoder_hidden_init = self.encoder.init_state()

        encoder_outputs, encoder_hidden = self.encoder(input_sequence, encoder_hidden_init)
        decoder_input = Variable(torch.LongTensor([SOS]*self.batch_size))

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.output_size))

        for i in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[i] = decoder_output
            decoder_input = target[i]
        #print(all_decoder_outputs.size(), target_batches.size())
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_batches.contiguous(),
            target_lengths
        )

        self.cnn_optimizer.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()
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
                label = [p.tolist() + [EOS] for p in label]
                target_lengths = [len(s) for s in label]
                target_padded = [self.pad_seq(s, max(target_lengths)) for s in label]
                target_padded = Variable(torch.LongTensor(target_padded))
                image = Variable(torch.FloatTensor(image))
                current_loss = self.train_step(image, target_padded, target_lengths)

                loss += current_loss
                if idx % self.print_step == 0:
                    print("EPOCH: %s BATCH: %s LOSS: %s AVG_LOSS: %s" % (epoch, idx/self.batch_size, current_loss, loss/(idx+1)))

            print("EPOCH: %s AVG_LOSS: %s" % (epoch, loss/size))

    def evaluate(self):
        pass


def main():

    batch_size = 8

    hyper_paramenters = {
        "learning_rate": 0.001,
        "batch_size": batch_size,
        "epoches": 100,
        "print_step": 2,
    }

    srn = SRN(config=hyper_paramenters, output_size=12, max_length=10)

    dataset = create_dataset(True)
    srn.train(dataset)


if __name__ == '__main__':

    main()