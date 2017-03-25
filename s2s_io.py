import dynet as dn
import random
from collections import defaultdict
from itertools import count
import sys


def init():
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    model = dn.Model()
    LAYERS = 6
    INPUT_DIM = 50
    HIDDEN_DIM = 50
    characters = list("abcdefghijklmnopqrstuvwxyz ,.(){};\n1234567890+-*/=<>!&|_")
    characters.append("<EOS>")
    characters.append("<SOS>")

    int2char = list(characters)
    char2int = {c:i for i,c in enumerate(characters)}

    VOCAB_SIZE = len(characters)
    params = {}
    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
    params["R"] = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
    params["C"] = model.add_parameters((HIDDEN_DIM, HIDDEN_DIM))
    params["bias"] = model.add_parameters((VOCAB_SIZE))
    


def sample_index(probs):
    rnd = random.random()
    for i,p in enumerate(probs):
        rnd -= p
        if rnd <= 0: break
    return i
def get_model():
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    lstm_encode = dn.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    lstm_decode = dn.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    return lstm_encode, lstm_decode
# return compute loss of RNN for one sentence
def do_one_sentence(rnn_enc, rnn_dec, in_sentence, out_sentence):
    from numpy import argmax
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    # setup the sentence
    dn.renew_cg()


    R = dn.parameter(params["R"])
    C = dn.parameter(params["C"])
    bias = dn.parameter(params["bias"])
    lookup = params["lookup"]
    in_sentence = ["<SOS>"] + list(in_sentence) + ["<EOS>"]
    out_sentence = ["<SOS>"] + list(out_sentence) + ["<EOS>"]
    s0 = rnn_enc.initial_state()
    s = s0
    for char in in_sentence:
        s = s.add_input(lookup[char2int[char]])
    s_enc = s
    loss = []
    s0 = rnn_dec.initial_state()
    s = s0
    for char, next_char in zip(out_sentence, out_sentence[1:]):
        s = s.add_input(lookup[char2int[char]])
        probs = dn.softmax(R*(s.output()+C*s_enc.output()) + bias)
        loss.append( -dn.log(dn.pick(probs,char2int[next_char])) )
        # loss.append( dn.pickneglogsoftmax(probs,char2int[char]) )
    loss = dn.esum(loss)
    return loss


# generate from model:
def generate(rnn_enc, rnn_dec, sentence):

    # setup the sentence
    dn.renew_cg()

    R = dn.parameter(params["R"])
    C = dn.parameter(params["C"])
    bias = dn.parameter(params["bias"])
    lookup = params["lookup"]
    sentence = ["<SOS>"] + list(sentence) + ["<EOS>"]
    s0 = rnn_enc.initial_state()
    s = s0
    ss = []
    for char in sentence:
        s = s.add_input(lookup[char2int[char]])
        ss.append(s)
    s_enc = s
    s0 = rnn_dec.initial_state()
    s = s0.add_input(lookup[char2int["<SOS>"]])
    out=[]
    while True:
        probs = dn.softmax(R*(s.output()+C*s_enc.output()) + bias)
        probs = probs.vec_value()
        next_char = sample_index(probs)
        out.append(int2char[next_char])
        if out[-1] == "<EOS>" or len(out)>20: break
        s = s.add_input(lookup[next_char])
    return "".join(out[:-1]) # strip the <EOS>


# train, and generate every 5 samples
def train(rnn_enc, rnn_dec, sentence_pairs, n_round=200):
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    trainer = dn.SimpleSGDTrainer(model)
    for i in xrange(n_round):
        if i % ((n_round+19)/20) == 0:
            n = len(sentence_pairs)
            idx = sample_index([1.0/n for i in range(n)])
        else:
            idx = -1
        for i, (in_s, out_s) in enumerate(sentence_pairs):
            out_s = in_s
            loss = do_one_sentence(rnn_enc, rnn_dec, in_s, out_s[::-1])
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            if i==idx:
                print loss_value, idx,
                print in_s,
                print generate(rnn_enc, rnn_dec, in_s)[::-1]

