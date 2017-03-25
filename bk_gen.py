
def gen_bracket(num=10, max_len=20, hor_p=0.7, vert_p=0.6):
    def deeper(nw_p, hor_p=hor_p, vert_p=vert_p):
        from random import random as rnd
        res = []
        p = 1.0
        nw_p *= vert_p
        if rnd()<=nw_p:
            while rnd()<=p:
                res += deeper(nw_p)
                p *= hor_p
        return ['(']+res+[')']
    res = []
    while len(res)<num:
        l = deeper(1.0)
        if len(l)<=max_len:
            res.append("".join(l))
    return res

def bracket2sentence(b):
    return " ".join(["l" if c=='(' else "r" for c in b])

def gen_data(num=10):
    return [(b, bracket2sentence(b)) for b in gen_bracket(num)]

