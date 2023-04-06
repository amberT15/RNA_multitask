import numpy as np

REVERSE_VOCAB = np.array(['A','C','G','T','N'])

def onehot_to_seq(onehot):
    adj = np.sum(onehot, axis=-1) == 0
    x_index = np.argmax(onehot,axis = -1) - adj
    seq_onehot = REVERSE_VOCAB[x_index]
    seq_char = [''.join(single_seq) for single_seq in seq_onehot]
    return seq_char

def seq_to_onehot(seq):
    seq_len = len(seq)
    seq_start = 0
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((4,seq_len), dtype='float16')

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == 'A':
                seq_code[0,i] = 1
            elif nt == 'C':
                seq_code[1,i] = 1
            elif nt == 'G':
                seq_code[2,i] = 1
            elif nt == 'T':
                seq_code[3,i] = 1
            else:
                seq_code[:,i] = 0.25

    return seq_code