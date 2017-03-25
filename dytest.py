import dynet as dn
import random
from collections import defaultdict
from itertools import count
import sys


def init():
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    model = dn.Model()
    LAYERS = 2
    INPUT_DIM = 50
    HIDDEN_DIM = 50
    characters = list("abcdefghijklmnopqrstuvwxyz ,.(){};\n1234567890+-*/=<>!&|_")
    characters.append("<EOS>")

    int2char = list(characters)
    char2int = {c:i for i,c in enumerate(characters)}

    VOCAB_SIZE = len(characters)
    params = {}
    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
    params["R"] = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
    params["bias"] = model.add_parameters((VOCAB_SIZE))
    


def get_model():
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    lstm = dn.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    srnn = dn.SimpleRNNBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    return lstm, srnn
# return compute loss of RNN for one sentence
def do_one_sentence(rnn, sentence):
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    # setup the sentence
    dn.renew_cg()
    s0 = rnn.initial_state()


    R = dn.parameter(params["R"])
    bias = dn.parameter(params["bias"])
    lookup = params["lookup"]
    sentence = ["<EOS>"] + list(sentence) + ["<EOS>"]
    sentence = [char2int[c] for c in sentence]
    s = s0
    loss = []
    for char,next_char in zip(sentence,sentence[1:]):
        s = s.add_input(lookup[char])
        probs = dn.softmax(R*s.output() + bias)
        loss.append( -dn.log(dn.pick(probs,next_char)) )
    loss = dn.esum(loss)
    return loss


# generate from model:
def generate(rnn):
    def sample(probs):
        rnd = random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    # setup the sentence
    dn.renew_cg()
    s0 = rnn.initial_state()

    R = dn.parameter(params["R"])
    bias = dn.parameter(params["bias"])
    lookup = params["lookup"]

    s = s0.add_input(lookup[char2int["<EOS>"]])
    out=[]
    while True:
        probs = dn.softmax(R*s.output() + bias)
        probs = probs.vec_value()
        next_char = sample(probs)
        out.append(int2char[next_char])
        if out[-1] == "<EOS>": break
        s = s.add_input(lookup[next_char])
    return "".join(out[:-1]) # strip the <EOS>


# train, and generate every 5 samples
def train(rnn, sentence, n_round=200):
    global characters, int2char, char2int, VOCAB_SIZE, model, params, LAYERS, INPUT_DIM, HIDDEN_DIM
    trainer = dn.SimpleSGDTrainer(model)
    for i in xrange(n_round):
        loss = do_one_sentence(rnn, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % (n_round/20) == 0:
            print loss_value,
            print generate(rnn)

