## self-compiled function collection for various data generations. (mt/saluki/annotation)

import glob
import json
import re
import math
import os
import sys
import h5py
import warnings
import six
import sequence
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
from kipoi.data import Dataset
from pybedtools import Interval
from pyfaidx import Fasta
from natsort import natsorted
import tensorflow as tf

# TFRecord constants
TFR_INPUT = 'sequence'
TFR_OUTPUT = 'target'

tissues = ['Retina - Eye', 'RPE/Choroid/Sclera - Eye', 'Subcutaneous - Adipose',
           'Visceral (Omentum) - Adipose', 'Adrenal Gland', 'Aorta - Artery',
           'Coronary - Artery', 'Tibial - Artery', 'Bladder', 'Amygdala - Brain',
           'Anterior cingulate - Brain', 'Caudate nucleus - Brain',
           'Cerebellar Hemisphere - Brain', 'Cerebellum - Brain', 'Cortex - Brain',
           'Frontal Cortex - Brain', 'Hippocampus - Brain', 'Hypothalamus - Brain',
           'Nucleus accumbens - Brain', 'Putamen - Brain',
           'Spinal cord (C1) - Brain', 'Substantia nigra - Brain',
           'Mammary Tissue - Breast', 'EBV-xform lymphocytes - Cells',
           'Leukemia (CML) - Cells', 'Xform. fibroblasts - Cells',
           'Ectocervix - Cervix', 'Endocervix - Cervix', 'Sigmoid - Colon',
           'Transverse - Colon', 'Gastroesoph. Junc. - Esophagus',
           'Mucosa - Esophagus', 'Muscularis - Esophagus', 'Fallopian Tube',
           'Atrial Appendage - Heart', 'Left Ventricle - Heart', 'Cortex - Kidney',
           'Liver', 'Lung', 'Minor Salivary Gland', 'Skeletal - Muscle',
           'Tibial - Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
           'Not Sun Exposed - Skin', 'Sun Exposed (Lower leg) - Skin',
           'Ileum - Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
           'Uterus', 'Vagina', 'Whole Blood']
bases = ['A', 'C', 'G', 'T']
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# # 0 unknown, 1 donor, 2 acceptor, 3 utr, 4 cds, 5 exon, -1 padding

def one_hot_encode(Xd, Yd,dim):
    task_count = int(np.max(Yd))
    OUT_MAP = np.zeros((dim+1,dim))
    for i in range(dim):
        OUT_MAP[i,i]=1
    return IN_MAP[Xd.astype('int8')], \
           OUT_MAP[Yd.astype('int8')]

def range_label(label_range,tx_start,tx_end,Y0,task_index):
    if isinstance(label_range, str):
        range_list = re.split(',',label_range)
        for r in range_list:
            r = re.split('-',r)
            start = int(r[0])
            end = int(r[1])
            Y0[start-tx_start:end-tx_start+1] = task_index
    return Y0

def create_datapoints(seq,row,seq_len,padding,task_list):
    #Sequence
    seq = seq.split('\t')[1][:-1]
    seq = 'N'*(padding//2) + str(seq) + 'N'*(padding//2)
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    tx_start = int(row['Start'])
    tx_end = int(row['End'])
    strand = row['Strand']
    #Task Splice
    donor = row['Donor']
    acceptor = row['Acceptor']
    if isinstance(donor, str) == False:
        jn_start =[]
        jn_end = []
    else:
        jn_start = np.array(re.split(',', donor)).astype(int)
        jn_end = np.array(re.split(',', acceptor)).astype(int)


    #Additional tasks
    Y0 = np.zeros(tx_end-tx_start+1)
    task_label = 3
    for task in task_list:
        target_range = row[task]
        Y0 = range_label(target_range,tx_start,tx_end,Y0,task_label)
        task_label += 1

    if strand == '+':
        X0 = np.asarray(list(seq),dtype='int')
        if len(jn_start) > 0:
            for c in jn_start:
                if tx_start <= c <= tx_end:
                    Y0[c-tx_start] = 1
            for c in jn_end:
                if tx_start <= c <= tx_end:
                    Y0[c-tx_start] = 2
    elif strand == '-':
        X0 = (5-np.asarray(list(seq[::-1]),dtype='int')) % 5  # Reverse complement
        if len(jn_start) > 0:
            for c in jn_end:
                if tx_start <= c <= tx_end:
                    Y0[tx_end-c] = 1
            for c in jn_start:
                if tx_start <= c <= tx_end:
                    Y0[tx_end-c] = 2

    Xd, Yd = reformat_data(X0, Y0, seq_len, padding)
    X, Y = one_hot_encode(Xd, Yd, task_label)

    return X,Y

def reformat_data(X0, Y0,SL,padding):
    num_points = int(math.ceil(len(Y0)/ SL))
    Xd = np.zeros((num_points, SL+padding))
    Yd = -np.ones((num_points, SL))

    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)
    Y0 = np.pad(Y0, [0, SL], 'constant', constant_values=-1)

    for i in range(num_points):
        Xd[i] = X0[SL*i:padding+SL*(i+1)]
    for i in range(num_points):
        Yd[i] = Y0[SL*i:SL*(i+1)]
    return Xd, Yd

def reformat_seq(X0,SL):
    num_points = int(math.ceil(len(X0)/ SL))
    Xd = []
    X0 = np.pad(X0, [0, SL], 'constant', constant_values='N')
    for i in range(num_points):
        Xd.append(''.join(X0[SL*i:SL*(i+1)]))
    return Xd

def onehot(seq):
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        if char == "N":
            pass
        else:
            X[i, bases.index(char.upper())] = 1
    return X

def logit(x):
    x = clip(x)
    return np.log(x) - np.log(1 - x)

def clip(x):
    return np.clip(x, 1e-5, 1-1e-5)

class Ascot(Dataset):
    ''' Load Acsot exons and PSI across tissues
        * Take the exon and flanking introns
        * Pad length to L.
        * If exon longer than L, cut out from the middle
        * If exon + flanking introns longer than L, cut the flanking intron
    '''

    def __init__(self,
                 ascot,
                 fasta_file,
                 length=900,
                 tissues=tissues,
                 encode=True,
                 pad_trim_same_l=True,
                 flanking=300,
                 flanking_exons=False,
                 region_anno=False,
                 seq_align='start',
                 mean_inpute=True,
                 use_logit=False):
        '''Args:
            - ascot: ascot psi file
            - fasta_file: fasta format file path
            - pad_trim_same_l: when True, trim sequence from middle of exon.
                           If false, return exon + flanking intron sequence of both sides
                           maxlen = flanking*2 + 300
            - flanking: length of intron to take, only effective when pad_trim_same_l=False
            - encode: whether encode sequence
            - region_anno: return binary indicator on region annotation
            - seq_align: if "both", return two sequence, one align from start, one from end
            - use_logit: if True, return target PSI in logits
            - flanking_exons: if True, return flanking exon sequence as well. e.g 100bp exon and 300bp intron
        '''

        if isinstance(fasta_file, six.string_types):
            fasta = Fasta(fasta_file, as_raw=False)
        self.fasta = fasta
        self.L = length
        self.tissues = tissues
        self.pad_trim_same_l = pad_trim_same_l
        self.encode = encode
        self.flanking = flanking
        if isinstance(flanking, tuple):
            self.flanking_l = flanking[0]
            self.flanking_r = flanking[0]
        self.exons, self.PSI, self.mean = self.read_exon(ascot)
        self.region_anno = region_anno
        self.seq_align = seq_align
        self.mean_inpute = mean_inpute
        if seq_align == 'both' and not encode:
            assert "When not encode sequence, only return one input sequence string"
        self.use_logit = use_logit
        self.flanking_exons = flanking_exons

    def __len__(self):
        return len(self.exons)

    def read_exon(self, ascot):
        exons = pd.read_csv(ascot, index_col=0)
        PSI = exons[self.tissues].values
        exons = exons[['chrom',
                       'exon_start',
                       'exon_end',
                       'intron_start',
                       'intron_end',
                       'strand',
                       'exon_id',
                       'gene_id']]
        PSI[PSI == -1] = np.nan
        m = np.nanmean(PSI, axis=1)
        m = m[:, np.newaxis]
        if np.mean(m) > 1:
            PSI = PSI / 100.
            m = m / 100.
        return exons, PSI, m

    def get_seq(self, exon):
        exon = ExonInterval(chrom=exon.chrom,
                            length=self.L,
                            start=exon.exon_start,
                            end=exon.exon_end,
                            strand=exon.strand,
                            intron_start=exon.intron_start,
                            intron_end=exon.intron_end,
                            fasta=self.fasta)
        return exon.sequence()

    def __getitem__(self, idx):
        exon = self.exons.iloc[idx]
        from copy import deepcopy
        psi = deepcopy(self.PSI[idx])
        m = self.mean[idx]

        # about sample weight:
        # (np.nanvar(psi) / np.squeeze(m)) * 100, (X)
        # std = (np.var(psi_copy) / np.squeeze(m)) * 100, psi_copy: mean inputed psi
        # np.var(psi_copy) * 100: OK, cor(sum_PSI_pred, NA frac) high

        # copy to compute var or std for sample weight
        psi_copy = deepcopy(psi)
        assert np.sum(psi_copy == -1.) == 0

        psi_copy[np.isnan(psi_copy)] = m
        std = min(np.var(psi_copy) * 100, 4)
        if self.mean_inpute:
            psi = psi_copy  # mean inpute

        out = {}
        if self.use_logit:
            psi = logit(psi)
        # convert back to -1
        #psi[np.isnan(psi)] = -1.
        out["targets"] = psi
        out["inputs"] = {}
        if self.pad_trim_same_l:
            seq = "N" * 50 + self.get_seq(exon) + "N" * 13
            if self.encode:
                seq = onehot(seq)
            out["inputs"]["seq"] = seq
        else:
            seq = self.fasta.get_seq(exon.chrom,
                                     exon.exon_start - self.flanking,
                                     exon.exon_end + self.flanking,
                                     exon.strand == '-')
            seq = seq.seq.upper()
            out['inputs']['fasta'] = seq
            if self.flanking_exons:
                if exon.strand == "+":
                    exon_up = self.fasta.get_seq(exon.chrom,
                                         exon.intron_start - self.L + self.flanking,
                                         exon.intron_start + self.flanking - 1,
                                         exon.strand == '-')
                    exon_up = exon_up.seq.upper()
                    exon_dw = self.fasta.get_seq(exon.chrom,
                                         exon.intron_end - self.flanking + 1,
                                         exon.intron_end + self.L - self.flanking,
                                         exon.strand == '-')
                    exon_dw = exon_dw.seq.upper()
                else:
                    exon_up = self.fasta.get_seq(exon.chrom,
                                         exon.intron_end - self.flanking + 1,
                                         exon.intron_end + self.L - self.flanking,
                                         exon.strand == '-')
                    exon_up = exon_up.seq.upper()
                    exon_dw = self.fasta.get_seq(exon.chrom,
                                         exon.intron_start - self.L + self.flanking,
                                         exon.intron_start + self.flanking - 1,
                                         exon.strand == '-')
                    exon_dw = exon_dw.seq.upper()

            if self.encode:
                # from mtsplice.utils.utils import HiddenPrints
                # with HiddenPrints():
                if self.seq_align == 'both':
                    seql = sequence.encodeDNA([seq], maxlen=self.L, seq_align='start')[0]
                    seqr = sequence.encodeDNA([seq], maxlen=self.L, seq_align='end')[0]
                    out["inputs"]["seql"] = seql
                    out["inputs"]["seqr"] = seqr
                    if self.flanking_exons:
                        out["inputs"]["exon_up"] = sequence.encodeDNA([exon_up])[0]
                        out["inputs"]["exon_dw"] = sequence.encodeDNA([exon_dw])[0]
                else:
                    seq = sequence.encodeDNA([seq], maxlen=self.L, seq_align=self.seq_align)[0]
                    out["inputs"]["seq"] = seq
            else:
                out["inputs"]["seq"] = seq
            if self.region_anno:
                anno = anno_region(self.flanking,
                                   exon.exon_end - exon.exon_start + 1,
                                   self.L, align=self.seq_align)
                out["inputs"]["anno"] = anno
        out["inputs"]["mean"] = np.repeat(logit(m), 56)
        out["inputs"]["std"] = std
        out['metadata'] = {}
        out['metadata']['chrom'] = exon.chrom
        out['metadata']['exon_id'] = exon.exon_id
        out['metadata']['exon_start'] = exon.exon_start
        out['metadata']['exon_end'] = exon.exon_end
        out['metadata']['intron_start'] = exon.intron_start
        out['metadata']['intron_end'] = exon.intron_end
        out['metadata']['strand'] = exon.strand
        return (out['inputs'],out['targets'])

class ExonInterval(Interval):
    ''' Encode exon logic
    '''

    def __init__(self,
                 length=600,
                 intron_start=None,
                 intron_end=None,
                 fasta=None,
                 **kwargs):
        ''' intron_start, intron_end are the start and end of the flanking introns of the given exon.
            intron_start is the start position on the left, intron_end is the end position on the right
        '''
        super().__init__(**kwargs)
        self.exon_start = self.start
        self.exon_end = self.end
        self.intron_start = intron_start
        self.intron_end = intron_end
        self.l = length
        self.fasta = fasta

    def getseq(self, start, end):
        seq = self.fasta.get_seq(self.chrom, start, end, self.strand == '-')
        ## TODO: pad or crop
        seq = seq.seq.upper()
        return seq

    def sequence(self):
        ''' Return padded or croped sequence with the same length
        '''
        exon_length = self.exon_end - self.exon_start + 1
        exon_intron_length = self.intron_end - self.intron_start + 1

        if exon_length > self.l:
            # start croping
            # + 63 because MMSplice takes 50 and 13 base in the intron
            crop_length = exon_length - self.l
            cutting_point = int((self.exon_end + self.exon_start) / 2)
            crop_left = int(crop_length / 2)
            crop_right = crop_length - crop_left
            if self.strand == "+":
                # add the required intron length
                crop_left += 50
                crop_right += 13
                seq_l = self.getseq(self.exon_start - 50, cutting_point - crop_left - 1)
                seq_r = self.getseq(cutting_point + crop_right, self.exon_end + 13)
                seq = seq_l + seq_r
            else:
                crop_left += 13
                crop_right += 50
                seq_l = self.getseq(self.exon_start - 13, cutting_point - crop_left - 1)
                seq_r = self.getseq(cutting_point + crop_right, self.exon_end + 50)
                seq = seq_r + seq_l
        elif exon_intron_length > self.l:
            crop_length = exon_intron_length - self.l
            # -2 to preserve dinucleotides
            if self.exon_start - self.intron_start < self.intron_end - self.exon_end:
                crop_left = min(int(crop_length / 2), self.exon_start - self.intron_start - 2)
                crop_right = crop_length - crop_left
                #assert crop_right > 0
            else:
                crop_right = min(int(crop_length / 2), self.intron_end - self.exon_end - 2)
                crop_left = crop_length - crop_right
                #assert crop_left > 0
            seq = self.getseq(self.intron_start + crop_left, self.intron_end - crop_right)

            # test
            _seq = self.getseq(self.exon_start, self.exon_end)
            if _seq in seq:
                warnings.warn("Seq does not contain the whole exon")

        else:
            pad_length = self.l - exon_intron_length
            pad_left = int(pad_length / 2)
            pad_right = pad_length - pad_left
            seq = self.getseq(self.intron_start, self.intron_end)
            seq = pad_left * "N" + seq + pad_right * "N"

            # test
            _seq = self.getseq(self.exon_start, self.exon_end)
            assert _seq in seq

        assert len(seq) == self.l
        return seq

def file_to_records(filename):
  return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

class RnaDataset:
  def __init__(self, data_dir, split_label, batch_size,
               mode='eval', shuffle_buffer=1024):
    """Initialize basic parameters; run make_dataset."""

    self.data_dir = data_dir
    self.batch_size = batch_size
    self.shuffle_buffer = shuffle_buffer
    self.mode = mode
    self.split_label = split_label

    # read data parameters
    data_stats_file = '%s/statistics.json' % self.data_dir
    with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
    self.length_t = data_stats['length_t']

    # self.seq_depth = data_stats.get('seq_depth',4)
    self.target_length = data_stats['target_length']
    self.num_targets = data_stats['num_targets']

    if self.split_label == '*':
      self.num_seqs = 0
      for dkey in data_stats:
        if dkey[-5:] == '_seqs':
          self.num_seqs += data_stats[dkey]
    else:
      self.num_seqs = data_stats['%s_seqs' % self.split_label]

    self.make_dataset()

  def batches_per_epoch(self):
    return self.num_seqs // self.batch_size

  def make_parser(self): #, rna_mode
    def parse_proto(example_protos):
      """Parse TFRecord protobuf."""

      feature_spec = {
        'lengths': tf.io.FixedLenFeature((1,), tf.int64),
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'coding': tf.io.FixedLenFeature([], tf.string),
        'splice': tf.io.FixedLenFeature([], tf.string),
        'targets': tf.io.FixedLenFeature([], tf.string)
      }

      # parse example into features
      feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

      # decode targets
      targets = tf.io.decode_raw(feature_tensors['targets'], tf.float16)
      targets = tf.cast(targets, tf.float32)

      # get length
      seq_lengths = feature_tensors['lengths']

      # decode sequence
      sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.uint8)
      sequence = tf.one_hot(sequence, 4)
      sequence = tf.cast(sequence, tf.float32)

      # decode coding frame
      coding = tf.io.decode_raw(feature_tensors['coding'], tf.uint8)
      coding = tf.expand_dims(coding, axis=1)
      coding = tf.cast(coding, tf.float32)

      # decode splice
      splice = tf.io.decode_raw(feature_tensors['splice'], tf.uint8)
      splice = tf.expand_dims(splice, axis=1)
      splice = tf.cast(splice, tf.float32)

      # concatenate input tracks
      inputs = tf.concat([sequence,coding,splice], axis=1)
      # inputs = tf.concat([sequence,splice], axis=1)
      # inputs = tf.concat([sequence,coding], axis=1)

      # pad to zeros to full length
      paddings = [[0, self.length_t-seq_lengths[0]],[0,0]]
      inputs = tf.pad(inputs, paddings)

      return inputs, targets

    return parse_proto

  def make_dataset(self, cycle_length=4):
    """Make Dataset w/ transformations."""

    # collect tfrecords
    tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, self.split_label)
    tfr_files = natsorted(glob.glob(tfr_path))

    # initialize tf.data
    if tfr_files:
      # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
      dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    else:
      print('Cannot order TFRecords %s' % tfr_path, file=sys.stderr)
      dataset = tf.data.Dataset.list_files(tfr_path)

    # train
    if self.mode == 'train':
      # repeat
      dataset = dataset.repeat()

      # interleave files
      dataset = dataset.interleave(map_func=file_to_records,
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # shuffle
      dataset = dataset.shuffle(buffer_size=self.shuffle_buffer,
        reshuffle_each_iteration=True)

    # valid/test
    else:
      # flat mix files
      dataset = dataset.flat_map(file_to_records)

    # map records to examples
    dataset = dataset.map(self.make_parser()) #self.rna_mode

    # batch
    dataset = dataset.batch(self.batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # hold on
    self.dataset = dataset


  def numpy(self, return_inputs=True, return_outputs=True):
    """ Convert TFR inputs and/or outputs to numpy arrays."""
    with tf.name_scope('numpy'):
      # initialize dataset from TFRecords glob
      tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, self.split_label)
      tfr_files = natsorted(glob.glob(tfr_path))
      if tfr_files:
        # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
      else:
        print('Cannot order TFRecords %s' % self.tfr_path, file=sys.stderr)
        dataset = tf.data.Dataset.list_files(self.tfr_path)

      # read TF Records
      dataset = dataset.flat_map(file_to_records)
      dataset = dataset.map(self.make_parser())
      dataset = dataset.batch(1)

    # initialize inputs and outputs
    seqs_1hot = []
    targets = []

    # collect inputs and outputs
    for seq_1hot, targets1 in dataset:
      # sequenceR
      if return_inputs:
        seqs_1hot.append(seq_1hot.numpy())

      # targets
      if return_outputs:
        targets.append(targets1.numpy())

    # make arrays
    seqs_1hot = np.array(seqs_1hot)
    targets = np.array(targets)

    # return
    if return_inputs and return_outputs:
      return seqs_1hot, targets
    elif return_inputs:
      return seqs_1hot
    else:
      return targets
