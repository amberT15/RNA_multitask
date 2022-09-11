import h5py
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append('/home/amber/multitask_RNA/DNA_BERT_rep/')
sys.path.append('/home/amber/multitask_RNA/')
from dna_tokenizer import DNATokenizer
from tqdm import tqdm

tokenizer = DNATokenizer('/home/amber/multitask_RNA/DNA_BERT_rep/vocab.txt')

spliceai_data = h5py.File('/home/amber/multitask_RNA/data/splice_ai/400/dataset_test_0.h5','r')
roberta_spliceai = h5py.File('/home/amber/multitask_RNA/data/splice_ai/roberta/roberta_input_test.h5','w')

vocab = np.array(['A','C','G','T','N'])
dataset_count = int(len(spliceai_data.keys())/2)
for i in range(0,dataset_count):
    roberta_input = []
    roberta_attention= []
    x_dataset = spliceai_data['X'+str(i)]
    adj = np.sum(x_dataset, axis=-1) == 0
    x_index = np.argmax(x_dataset,axis = -1) - adj
    seq_onehot = vocab[x_index]
    seq_char = [''.join(single_seq) for single_seq in seq_onehot]
    for s in tqdm(range(0,len(seq_char))):
        seq = seq_char[s]
        split_seq = np.array([seq[i:i+6] for i in range(0, len(seq)-6+1, 1)])
        flag = ['N' in kmer for kmer in split_seq]
        split_seq = np.where(flag,'[PAD]',split_seq)
        split_seq = ' '.join(split_seq)
        token_seq = tokenizer.batch_encode_plus([split_seq],max_length = 507, 
                            context_split=True,
                            return_tensors = 'pt',
                            add_special_tokens=True,
                            return_attention_masks=True)
        roberta_input.extend(token_seq['input_ids'].numpy())
        roberta_attention.extend(token_seq['attention_mask'].numpy())

    roberta_spliceai.create_dataset(name='X'+str(i),data=np.array(roberta_input))
    roberta_spliceai.create_dataset(name='A'+str(i),data=np.array(roberta_attention))

spliceai_data.close()
roberta_spliceai.close()