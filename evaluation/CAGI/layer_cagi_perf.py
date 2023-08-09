# %%
import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
import scipy.stats
import numpy as np
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm
import math
import h5py
import sys
sys.path.append('/home/ztang/multitask_RNA/data_generation')
import utils 
import tensorflow as tf
### GPU memory usage settings
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#### nucleotide transformer model choice
model_name = '2B5_1000G'
if '2B5' in model_name:
    print('2B5_model')
    max_layer = 32
else:
    print('500M model')
    max_layer = 24

datalen = 230
cell_name = 'K562'

##### Read CAGI dataset
file = h5py.File("/home/ztang/multitask_RNA/data/CAGI/"+cell_name+"/onehot.h5", "r")
alt = file['alt']
ref = file['ref']

#### Read CAGI metadata
exp_df = pd.read_csv('/home/ztang/multitask_RNA/data/CAGI/'+cell_name+'/metadata.csv')
target = exp_df['6'].values.tolist()
exp_perf = {}
for exp in exp_df['8'].unique():
   exp_perf[exp] = []

#### initilize experiment
random_key = jax.random.PRNGKey(0)
N, L, A = alt.shape
max_len = math.ceil(datalen/6)+2
batch_size = 1024

for embed_layer in range(1,max_layer+1):
#for embed_layer in range(1,2):
    #### Load model with selected embedding output layer
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        mixed_precision=False,
        embeddings_layers_to_save=(embed_layer,),
        attention_maps_to_save=(),
        max_positions=max_len,
    )
    forward_fn = hk.transform(forward_fn)
    #### Load corresponding embedding trained model
    model_dir = '/home/ztang/multitask_RNA/model/lenti_MPRA_embed/'+cell_name+'/layer_'+str(embed_layer)+'.h5'
    model = tf.keras.models.load_model(model_dir)
    pred_diff = []

    #### for each batch
    for i in tqdm(range(0,N,batch_size)):
        b_size = batch_size
        if i + batch_size > N:
            b_size = N-i
        onehot = np.concatenate((ref[i:i+b_size],alt[i:i+b_size]))
        seq = utils.onehot_to_seq(onehot)
        token_out = tokenizer.batch_tokenize(seq)
        token_id = [b[1] for b in token_out]
        seq_pair = jnp.asarray(token_id,dtype=jnp.int32)
        outs = forward_fn.apply(parameters, random_key, seq_pair)
        embed = np.array(outs['embeddings_'+str(embed_layer)])
    
        ### Make prediction according to embeddings    
        embed_pred = model.predict(embed)
        ### Add prediction difference to list
        pred_diff.extend(embed_pred[b_size:2*b_size] - embed_pred[0:b_size])
    
    #### calculate pearson per experiment for all results
    for exp in exp_df['8'].unique():
        sub_df = exp_df[exp_df['8'] == exp]
        exp_target = np.array(target)[sub_df.index.to_list()]
        exp_pred = np.squeeze(pred_diff)[sub_df.index.to_list()]
        exp_perf[exp].extend([scipy.stats.pearsonr(exp_pred,exp_target)[0]])

#### export results to csv
perf_df = pd.DataFrame.from_dict(exp_perf)
perf_df.index.name = 'Layer'
perf_df.to_csv('./result/'+cell_name+'_layer_model.csv')
    


        

