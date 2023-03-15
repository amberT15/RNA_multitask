import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import math
import os
import util
import random
import h5py
import subprocess 
import sys

species = sys.argv[1]
seq_length = int(sys.argv[2])

data_dir = '/home/amber/multitask_RNA/data/species/' + species
annotation_file = glob.glob(data_dir + '/*.gtf')[0]
reference_genome = glob.glob(data_dir + '/*.fa')[0]
output_fa = data_dir+'/selected_seq.fa'
bed_file = data_dir+'/selected_seq.bed'
h5_file = data_dir+'/'+str(seq_length)+'_onehot.h5'

colnames=['Chrom', 'Database', 'Annotation', 'Start','End','Score','Strand','Phase','Notes'] 
annotation_df = pd.read_csv(annotation_file,sep='\t',skiprows=5,names=colnames,header=None)
annotation_df = annotation_df.drop('Phase', 1)
annotation_df = annotation_df.drop('Score', 1)
annotation_df['GeneID'] = annotation_df['Notes'].apply(lambda x: x.split('"')[1])
annotation_df['TranscriptID'] = annotation_df['Notes'].apply(lambda x : x.split('"')[3])
trans_id_list = []
gene_list = annotation_df['GeneID'].unique()
all_transcript_df = annotation_df[annotation_df['Annotation'] == 'transcript']

for gene in tqdm(gene_list,total=len(gene_list)):
    transcript_df = all_transcript_df[all_transcript_df['GeneID']==gene]
    trans_length = np.absolute(transcript_df['End'] - transcript_df['Start'])
    max_trans_index = np.argmax(trans_length)
    max_trans = transcript_df.iloc[max_trans_index]
    max_id = max_trans['Notes'].split('"')[3]
    trans_id_list.append(max_id)
transcript_flag = (all_transcript_df['TranscriptID'].isin(trans_id_list))
cannonical_df = all_transcript_df[transcript_flag]
cannonical_df.to_csv(data_dir + '/cannonical_transcript.csv',sep='\t',
                     columns=['Chrom', 'Database', 'Annotation', 'Start','End','Strand','GeneID','TranscriptID'])

chrom = []
strand = []
start_l = []
end_l = []
gene = []
for index,row in tqdm(cannonical_df.iterrows()):
    start = row['Start']
    end = row['End']
    length = end - start
    split_count = int(math.ceil((end-start)/seq_length))
    total_length = split_count * seq_length
    pad_length = (total_length - length)/2
    #pad toward both side unless exceeds genome size (how to judge chrom size)
    start = start - math.ceil(pad_length)
    end = end + math.floor(pad_length)
    if start < 0:
        start = 0
        end = end - start
    split = [(round(seq_length*i)+start, round(seq_length*(i+1))+start) for i in range(split_count)]
    for entry in split:
        chrom.append(row['Chrom'])
        strand.append(row['Strand'])
        start_l.append(entry[0])
        end_l.append(entry[1])
        gene.append(row['GeneID'])

bed_df = pd.DataFrame({'Chr':chrom,'Start':start_l,'End':end_l,'Strand':strand,'Gene':gene})
bed_df = bed_df.sort_values(by=['Chr','Gene','Start'])

bed_df = bed_df[['Chr','Start','End','Gene','Gene','Strand']]
if ('chr' in str(bed_df['Chr'][0])) == False:
    bed_df['Chr'] = 'chr' + bed_df['Chr'].astype(str)
bed_df.to_csv( bed_file,
                index=False,header=False,sep = '\t')
cmd = ['bedtools', 'getfasta', '-fi', reference_genome, '-bed', bed_file, '-fo', output_fa]
subprocess.Popen(cmd).wait()

fasta = open(output_fa, 'r')
lines = fasta.readlines()
seq = []
# Strips the newline character
for line in tqdm(lines[1::2]):
    if line[0] == '>':
        print('error in line count')
        break
    else:
        seq.append(line.strip().upper())
random.shuffle(seq)
data_length = len(seq)
train_data = seq[:int(data_length*0.9)]
valid_data = seq[int(data_length*0.9):]
valid_code = []
for seq in valid_data:
    valid_code.append(util.seq_to_onehot(seq))
train_code = []
for seq in train_data:
    train_code.append(util.seq_to_onehot(seq))
    
h5f = h5py.File(h5_file, 'w')
h5f.create_dataset('train',data = train_code)
h5f.create_dataset('valid',data = valid_code)
h5f.close()
