CL_max=400
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

splice_table='canonical_dataset.txt'
ref_genome='/home/amber/ref/hg19/hg19.fa'
# Input details

data_dir='/home/amber/multitask_RNA/data/splice_ai/400/'
sequence='canonical_sequence.txt'
# Output details
