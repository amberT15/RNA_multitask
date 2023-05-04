# %%
import h5py
import sys
sys.path.append('/home/ztang/multitask_RNA/data_generation')
import utils
import numpy as np

# %%
file = h5py.File("/home/ztang/multitask_RNA/data/CAGI/CAGI_onehot.h5", "r")
alt = file['alt']
ref = file['ref']

# %% [markdown]
# ## Nucleotide Transformer zero shot test

# %% [markdown]
# cosine similarity between embeddings with different allele

# %%
import nucleotide_transformer
import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm
model_name = '2B5_1000G'
if '2B5' in model_name:
    print('2B5_model')
    embed_layer = 32
else:
    print('500M model')
    embed_layer = 24

# %%
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    mixed_precision=False,
    embeddings_layers_to_save=(embed_layer,),
    attention_maps_to_save=(),
    max_positions=513,
)
forward_fn = hk.transform(forward_fn)

# %%
random_key = jax.random.PRNGKey(0)
N, L, A = alt.shape
mut_i = int(L/2-1)
batch_size = 128
cagi_llr=[]
for i in tqdm(range(0,N,batch_size)):
    b_size = 128
    if i + batch_size > N:
        b_size = N-batch_size
    onehot = np.concatenate((ref[i:i+b_size],alt[i:i+b_size]))
    seq = utils.onehot_to_seq(onehot)
    token_out = tokenizer.batch_tokenize(seq)
    token_id = [b[1] for b in token_out]
    seq_pair = jnp.asarray(token_id,dtype=jnp.int32)
    outs = forward_fn.apply(parameters, random_key, seq_pair)
    for i in range(b_size):
        ref_out = outs['embeddings_'+str(embed_layer)][i]
        alt_out = outs['embeddings_'+str(embed_layer)][i+b_size]
        cagi_llr.append((ref_out * alt_out).sum()/(jnp.linalg.norm(ref_out)*jnp.linalg.norm(alt_out)))


# %%
output = h5py.File('/home/ztang/multitask_RNA/data/CAGI/zero_shot/cagi_'+model_name+'.h5', 'w')
output.create_dataset('llr', data=np.array(cagi_llr))
output.close()
