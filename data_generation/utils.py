import numpy as np

REVERSE_VOCAB = np.array(['A','C','G','T','N'])

def onehot_to_seq(onehot):
    adj = np.sum(onehot, axis=-1) == 0
    x_index = np.argmax(onehot,axis = -1) - adj
    seq_onehot = REVERSE_VOCAB[x_index]
    seq_char = [''.join(single_seq) for single_seq in seq_onehot]
    return seq_char
