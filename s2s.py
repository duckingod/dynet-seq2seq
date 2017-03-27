#!/usr/bin/env python
import dynet as dn
import random
from collections import defaultdict
from itertools import count
import sys

BEG_SEQ = "<S>"
END_SEQ = "</S>"

def sample_index(probs):
    rnd = random.random()
    for i,p in enumerate(probs):
        rnd -= p
        if rnd <= 0: break
    return i

# input all thing to lstm
# return (final state, list of all states)
def input_all(lstm_state, input_list):
    s = lstm_state
    ss = [s]
    for i in input_list:
        s = s.add_input(i)
        ss.append(s)
    return s, ss

class BasicModel(object):
    def __init__(self):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ,.(){};\n1234567890+-*/=<>!&|_")+[BEG_SEQ, END_SEQ]
        self.int2char = self.characters
        self.char2int = {c:i for i,c in enumerate(self.int2char)}
        self.vocab_size = len(self.int2char)
    def wrap_sentence(self, s):
        return [ self.char2int[c] for c in [BEG_SEQ]+list(s)+[END_SEQ] ]

class Seq2Seq(BasicModel):
    # input dim != hidden dim cause dynet's bug
    def __init__(self, n_layers=3, input_dim=13, hidden_dim=13):
        super(self.__class__, self).__init__()
        self.layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = dn.Model()
        self.p = {}
        p = self.p
        model = self.model
        p["lookup"] = model.add_lookup_parameters((self.vocab_size, self.input_dim))
        p["R"] = model.add_parameters((self.vocab_size, self.hidden_dim))
        p["C"] = model.add_parameters((self.vocab_size, self.hidden_dim))
        p["bias"] = model.add_parameters((self.vocab_size))
        self.encoder = dn.LSTMBuilder(self.layers, self.input_dim, self.hidden_dim, model)
        self.decoder = dn.LSTMBuilder(self.layers, self.input_dim, self.hidden_dim, model)
    # lookup, R, C, bias, encoder, decoder = self.get_params()
    def get_params(self):
        f = dn.parameter
        p = self.p
        return p['lookup'], f(p['R']), f(p['C']), f(p['bias']), self.encoder, self.decoder
    # return compute loss of RNN for one sentence
    def compute_loss(self, in_sentence, out_sentence):
        from numpy import argmax
        from dynet import dropout
        dn.renew_cg()
        lookup, R, C, bias, encoder, decoder = self.get_params()
        in_s, out_s = self.wrap_sentence(in_sentence), self.wrap_sentence(out_sentence)

        loss = []
        enc_s, _ = input_all(encoder.initial_state(), [lookup[c] for c in in_s])
        s = decoder.initial_state().add_input(enc_s.output())
        for char, next_char in zip(out_s, out_s[1:]):
            s = s.add_input(lookup[char])
            probs = dn.softmax(R*s.output() + bias)
            loss.append( -dn.log(dn.pick(probs,next_char)) )
            # loss.append( dn.pickneglogsoftmax(probs,next_char) )
        loss = dn.esum(loss)
        return loss
    # generate from model:
    def generate(self, sentence):
        dn.renew_cg()

        lookup, R, C, bias, encoder, decoder = self.get_params()
        sentence = self.wrap_sentence(sentence)
        enc_s, _ = input_all(encoder.initial_state(), [lookup[c] for c in sentence])
        s = decoder.initial_state().add_input(enc_s.output())
        out=[]
        while True:
            probs = dn.softmax(R*s.output() + bias)
            next_char = sample_index(probs.vec_value())
            out.append(self.int2char[next_char])
            if out[-1] == END_SEQ or len(out)>20: break
            s = s.add_input(lookup[next_char])
        return "".join(out[:-1]) # strip the <EOS>

class Seq2SeqAttn(Seq2Seq):
    def __init__(self, n_layers=4, input_dim=50, hidden_dim=50, attn_dim=50):
        super(self.__class__, self).__init__()
        self.layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = dn.Model()
        self.p = {}
        p = self.p
        model = self.model
        p["lookup"] = model.add_lookup_parameters((self.vocab_size, self.input_dim))
        p["R"] = model.add_parameters((self.vocab_size, self.hidden_dim))
        p["C"] = model.add_parameters((self.vocab_size, self.hidden_dim))
        p["bias"] = model.add_parameters((self.vocab_size))
        self.encoder = dn.LSTMBuilder(self.layers, self.input_dim, self.hidden_dim, model)
        self.decoder = dn.LSTMBuilder(self.layers, self.input_dim, self.hidden_dim, model)
    # lookup, R, C, bias, encoder, decoder = self.get_params()
    def get_params(self):
        f = dn.parameter
        p = self.p
        return p['lookup'], f(p['R']), f(p['C']), f(p['bias']), self.encoder, self.decoder
    # return compute loss of RNN for one sentence
    def compute_loss(self, in_sentence, out_sentence):
        from numpy import argmax
        dn.renew_cg()
        lookup, R, C, bias, encoder, decoder = self.get_params()
        in_s, out_s = self.wrap_sentence(in_sentence), self.wrap_sentence(out_sentence)

        loss = []
        enc_s, _ = input_all(encoder.initial_state(), [lookup[c] for c in in_s])
        s = decoder.initial_state().add_input(enc_s.output())
        for char, next_char in zip(out_s, out_s[1:]):
            s = s.add_input(lookup[char])
            probs = dn.softmax(R*s.output() + bias)
            loss.append( -dn.log(dn.pick(probs,next_char)) )
            # loss.append( dn.pickneglogsoftmax(probs,next_char) )
        loss = dn.esum(loss)
        return loss
    # generate from model:
    def generate(self, sentence):
        dn.renew_cg()

        lookup, R, C, bias, encoder, decoder = self.get_params()
        sentence = self.wrap_sentence(sentence)
        enc_s, _ = input_all(encoder.initial_state(), [lookup[c] for c in sentence])
        s = decoder.initial_state().add_input(enc_s.output())
        out=[]
        while True:
            probs = dn.softmax(R*s.output() + bias)
            next_char = sample_index(probs.vec_value())
            out.append(self.int2char[next_char])
            if out[-1] == END_SEQ or len(out)>20: break
            s = s.add_input(lookup[next_char])
        return "".join(out[:-1]) # strip the <EOS>

# train, and generate every 5% samples
def train(seq2seq, sentence_pairs, n_round=200):
    trainer = dn.SimpleSGDTrainer(seq2seq.model)
    for i in xrange(n_round):
        if (i+1) % ((n_round+19)/20) == 0:
            from random import randint
            idx = randint(0, len(sentence_pairs)-1)
        else:
            idx = -1
        for i, (in_s, out_s) in enumerate(sentence_pairs):
            loss = seq2seq.compute_loss(in_s, out_s[::-1])
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            if i==idx:
                print loss_value, idx,
                print in_s, " >>> ", 
                print seq2seq.generate(in_s)[::-1]


if __name__=="__main__":
    s2s = Seq2Seq()
    train(s2s, [("who are u", "hi i m belle"), ("what is that", "an apple")], 1000)




