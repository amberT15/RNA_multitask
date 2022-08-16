import h5py
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append('/home/amber/multitask_RNA/DNA_BERT_rep/')
sys.path.append('/home/amber/multitask_RNA/')
import rna_model
import transformers
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from dnabert_datastruct import DNATokenizer,rnabert_maskwrapper
from tqdm import tqdm
length = 510

model = RobertaForMaskedLM.from_pretrained('/home/amber/multitask_RNA/DNA_BERT_rep/small-roberta-lr8/checkpoint-23500/')
tokenizer = DNATokenizer.from_pretrained('dna6')

spliceai_data = h5py.File('/home/amber/multitask_RNA/data/splice_ai/400/dataset_train_all.h5','r')
roberta_spliceai = h5py.File('/home/amber/multitask_RNA/data/splice_ai/roberta/roberat_input.h5','w')

vocab = np.array(['A','C','G','T','N'])
dataset_count = int(len(spliceai_data.keys())/2)
for i in tqdm(range(0,dataset_count)):
    roberta_input = []

    x_dataset = spliceai_data['X'+str(i)]
    adj = np.sum(x_dataset, axis=-1) == 0
    x_index = np.argmax(x_dataset,axis = -1) - adj
    seq_onehot = vocab[x_index]
    seq_char = [''.join(single_seq) for single_seq in seq_onehot]
    for seq in tqdm(seq_char):
        chop_seq = [seq[i:min([i+length,len(seq)])] for i in range(0,len(seq),length)]
        split_seq = [' '.join(short_seq[i:i+6] for i in range(0, len(short_seq)-6+1, 1)) for short_seq in chop_seq]
        encoded_batch = tokenizer.batch_encode_plus(split_seq,add_special_tokens=True,return_tensors='pt')['input_ids']
        roberta_input.extend(encoded_batch)
    roberta_spliceai.create_dataset(name='X'+str[i],data=roberta_input)