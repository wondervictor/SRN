# -*- coding: utf-8 -*-


import torch

class Beam(object):

    def __init__(self, beam_size, vocabulary):
        self.beam_size = beam_size
        self.pad = vocabulary['<pad>']
        self.bos = vocabulary['<bos>']
        self.eos = vocabulary['<eos>']

        self.scores = torch.zeros(self.beam_size)

        self.prevs = []
        self.nexts = [torch.LongTensor(self.beam_size).fill_(self.pad)]
        self.attn = []
        self.nexts[0][0] = self.bos


    def get_current_state(self):
        return
