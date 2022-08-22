import h5py
import numpy as np
import torch
from transformers import RobertaForMaskedLM
from tqdm import tqdm
import sys

start = sys.argv[1]
end = sys.argv[2]

model = RobertaForMaskedLM.from_pretrained('/home/amber/multitask_RNA/DNA_BERT_rep/small-roberta-lr8/checkpoint-23500/')
file = h5py.File('/home/amber/multitask_RNA/data/splice_ai/roberta/roberta_input.h5','r')

roberta_output = h5py.File('/home/amber/multitask_RNA/data/splice_ai/roberta/roberat_output_'+str(start)+'_'+str(end)+'.h5','w')

for i in range(int(start),int(end)):
    input_set = file['X'+str(i)]
    seq_cache = []
    for seq_i in tqdm(range(0,len(input_set),11)):
        seq_batch = torch.tensor(input_set[seq_i:seq_i+11])
        output_seq = model.roberta.embeddings(seq_batch).detach().numpy()
        concat_seq = np.concatenate(output_seq[0:10,1:506,:],axis = 0)
        concat_seq = np.concatenate((concat_seq,output_seq[-1,1:346,:]),axis = 0)
        seq_cache.append(concat_seq)

    roberta_output.create_dataset(name='X'+str(i),data=np.array(seq_cache))

file.close()
roberta_output.close()