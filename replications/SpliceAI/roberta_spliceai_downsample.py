import numpy as np
import sys
import time
import h5py
import wandb
import keras.backend as kb
import tensorflow as tf
from spliceai import *
from utils import *
from constants import * 
import os

data_ratio = float(sys.argv[1])
model_save_path = str(sys.argv[2])

wandb.init(entity='ambert',project="spliceai_downsample",
        config={'model':'Roberta','ratio':data_ratio})

###############################################################################
# Model
###############################################################################

L = 32
strategy = tf.distribute.MirroredStrategy()
N_GPUS = strategy.num_replicas_in_sync

W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11],dtype=np.int)
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4],dtype=np.int)
BATCH_SIZE = 18*N_GPUS

CL = 2 * np.sum(AR*(W-1))
assert CL <= CL_max and CL == int(400)
print ("\033[1mContext nucleotides: %d\033[0m" % (CL))
print ("\033[1mSequence length (output): %d\033[0m" % (SL))
print ("\033[1mBatch size and GPU usage: %d * %d\033[0m" % (BATCH_SIZE/N_GPUS,N_GPUS))

with strategy.scope():  
    model_m = RobertaAI(L, W, AR)
    model_m.compile(loss=categorical_crossentropy_2d,optimizer='adam')

input_f = h5py.File('../../data/splice_ai/roberta/roberta_output.h5','r')
target_f = h5py.File('../../data/splice_ai/400/dataset_train_all.h5','r')

num_idx = len(target_f.keys())//2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.9*data_ratio*num_idx)]
idx_valid = idx_all[int(0.9*num_idx):]

EPOCH_NUM = 10*len(idx_train)

start_time = time.time()


for epoch_num in range(EPOCH_NUM):

    idx = np.random.choice(idx_train)

    X = input_f['X' + str(idx)]
    Y = target_f['Y' + str(idx)]
    model_m.fit(X, Y[0], batch_size=BATCH_SIZE, verbose=0,shuffle=False)


    if (epoch_num+1) % len(idx_train) == 0:
        # Printing metrics (see utils.py for details)

        print ("--------------------------------------------------------------")
        print ("\n\033[1mValidation set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_valid:

            X = input_f['X' + str(idx)]
            Y = target_f['Y' + str(idx)]

            Yp = model_m.predict(X, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):

                is_expr = (Y[t].sum(axis=(1,2)) >= 1)

                Y_true_1[t].extend(Y[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Y[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

        print ("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            top_k,test_aupr = print_topl_statistics(np.asarray(Y_true_1[t]),
                                  np.asarray(Y_pred_1[t]))
            if test_aupr <= 0.001:
                wandb.finish(exit_code=1)
                print ("\n\033[1mFailed initilization. Re-starting training.\033[0m")
                os.execv(sys.executable, ['python'] + [sys.argv[0]] + [sys.argv[1]] + [sys.argv[2]])

            wandb.log({"acceptor_top_k":top_k,"acceptor_aupr":test_aupr})

        print ("\n\033[1mDonor:\033[0m")
        for t in range(1):
            d1,d2 =print_topl_statistics(np.asarray(Y_true_2[t]),
                                  np.asarray(Y_pred_2[t]))

            wandb.log({"donor_top_k":d1,"donor_aupr":d2})
            
        print ("\n\033[1mTraining set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_train[:len(idx_valid)]:

            X = input_f['X' + str(idx)]
            Y = target_f['Y' + str(idx)]

            Yp = model_m.predict(X, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):

                is_expr = (Y[t].sum(axis=(1,2)) >= 1)

                Y_true_1[t].extend(Y[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Y[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

        print ("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            d1,d2=print_topl_statistics(np.asarray(Y_true_1[t]),
                                  np.asarray(Y_pred_1[t]))

        print ("\n\033[1mDonor:\033[0m")
        for t in range(1):
            d1,d2 =print_topl_statistics(np.asarray(Y_true_2[t]),
                                  np.asarray(Y_pred_2[t]))

        print ("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
        print ("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        print ("--------------------------------------------------------------")

        model_m.save(model_save_path)

        if (epoch_num+1) >= 6*len(idx_train):
            kb.set_value(model_m.optimizer.lr,
                         0.5*kb.get_value(model_m.optimizer.lr))
            # Learning rate decay

input_f.close()
target_f.close()
        
###############################################################################