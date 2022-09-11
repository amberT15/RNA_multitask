from mt_layers import SplineWeight1D,mask_kl
import tensorflow as tf
from tensorflow import keras
import os
import h5py 
import sys
sys.path.append('./gopher/gopher/')
import modelzoo
import numpy as npç
import wandb
from wandb.keras import WandbCallback

def MTsplice(input_size,initializer):  
    seql = tf.keras.layers.Input(shape=input_size)
    seqr = tf.keras.layers.Input(shape=input_size)
    convl = tf.keras.layers.Conv1D(64,11,padding='same',activation='relu',
                                    kernel_initializer=initializer)(seql)
    convr = tf.keras.layers.Conv1D(64,11,padding='same',activation='relu',
                                    kernel_initializer=initializer)(seqr)

    for i in range(8):
        block_l = keras.layers.BatchNormalization()(convl)
        block_l = keras.layers.Conv1D(64,3,padding='same',dilation_rate = 2**(i+1),activation='relu',
                                        kernel_initializer=initializer )(block_l)
        block_r = keras.layers.BatchNormalization()(convr)
        block_r = keras.layers.Conv1D(64,3,padding='same',dilation_rate = 2**(i+1),activation='relu',
                                        kernel_initializer=initializer)(block_r)

        convl = keras.layers.Add()([block_l, convl])
        convr = keras.layers.Add()([block_r, convr])
    
    l_branch = SplineWeight1D(l2_smooth = 0.001)(convl)
    r_branch = SplineWeight1D(l2_smooth = 0.001)(convr)

    cat = keras.layers.Concatenate(-2)([l_branch,r_branch])
    pool = keras.layers.GlobalAveragePooling1D()(cat)
    pool = keras.layers.BatchNormalization()(pool)
    dense = keras.layers.Dense(32,activation='relu',kernel_initializer=initializer)(pool)
    dense = keras.layers.BatchNormalization()(dense)
    dropout = keras.layers.Dropout(0.5)(dense)
    output = keras.layers.Dense(56,kernel_initializer=initializer)(dropout)

    model = tf.keras.models.Model(inputs=[seql,seqr], outputs=output)

    return model

def train_model(config):

    f = h5py.File('./data/MT_Splice/psi_data_logit.h5','r')
    seql_train = f['seql_train']
    seqr_train = f['seqr_train']
    y_train = f['y_train']

    seql_valid = f['seql_valid']
    seqr_valid = f['seqr_valid']
    y_valid = f['y_valid']

    init = tf.keras.initializers.VarianceScaling(
                                                scale=2,
                                                mode='fan_in',
                                                distribution='normal')

    #Training script
    model = MTsplice(config['input_size'],init)
    model.compile(tf.keras.optimizers.Adam(config['lr']),
            loss = mask_kl())

    model.fit(
        x = (seql_train,seqr_train),y = y_train,
        epochs= config['max_epoch'],
        batch_size = config['batch_size'],
        validation_data = ((seql_valid,seqr_valid),y_valid),
        shuffle = False,
        callbacks=[modelzoo.early_stopping(config['es_patience']),
                modelzoo.reduce_lr(patience = config['lr_patience']),
                WandbCallback()]
        )
    f.close()

def train_config(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        history = train_model(config=config)

def main():
    if len(sys.argv) > 1:
        exp_id = sys.argv[1]
        exp_n = sys.argv[2]
        if 'sweeps' in exp_id:
            exp_id = '/'.join(exp_id.split('/sweeps/'))
        else:
            raise BaseException('Sweep ID invalid!')
        sweep_id = exp_id
        wandb.login()
        wandb.agent(sweep_id, train_config, count=exp_n)
    else:
        print('training with default')
        train_config = {
            'lr':0.001,
            'es_patience':5,
            'lr_patience':3,
            'batch_size':128,
            'max_epoch':100,
            'input_size':(400,4)
        }
        wandb.login()
        wandb.init(project="mtsplice_replicate", 
                config = train_config)
        train_model(config = train_config)

# __main__
################################################################################
if __name__ == '__main__':
    main()
