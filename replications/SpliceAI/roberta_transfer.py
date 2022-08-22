import numpy as np
import sys
import time
import h5py
import keras.backend as kb
import tensorflow as tf
from spliceai import *
from utils import *
from multi_gpu import *
from constants import * 
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'


###############################################################################
# Model
###############################################################################

L = 32
N_GPUS = 1

W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11],dtype=np.int)
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4],dtype=np.int)
BATCH_SIZE = 1*N_GPUS


CL = 2 * np.sum(AR*(W-1))
assert CL <= CL_max and CL == int(400)
print ("\033[1mContext nucleotides: %d\033[0m" % (CL))
print ("\033[1mSequence length (output): %d\033[0m" % (SL))

model = RobertaAI(L, W, AR)
model.summary()
model_m = make_parallel(model, N_GPUS)
model_m.compile(loss=categorical_crossentropy_2d, optimizer='adam')


# h5f = h5py.File(data_dir + 'dataset' + '_' + 'train'
#                 + '_' + 'all' + '.h5', 'r')

input_f = h5py.File('/home/amber/multitask_RNA/data/splice_ai/roberta/roberta_output.h5','r')
target_f = h5py.File('/home/amber/multitask_RNA/data/splice_ai/400/dataset_train_all.h5','r')

num_idx = len(target_f.keys())//2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.9*num_idx)]
idx_valid = idx_all[int(0.9*num_idx):]

EPOCH_NUM = 10*len(idx_train)

start_time = time.time()


for epoch_num in range(EPOCH_NUM):

    idx = np.random.choice(idx_train)

    X = input_f['X' + str(idx)][:]
    Y = target_f['Y' + str(idx)][:]

    Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 
    model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=1)


    if (epoch_num+1) % len(idx_train) == 0:
        # Printing metrics (see utils.py for details)

        print ("--------------------------------------------------------------")
        print ("\n\033[1mValidation set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_valid:

            X = input_f['X' + str(idx)][:]
            Y = target_f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):

                is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

        print ("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_1[t]),
                                  np.asarray(Y_pred_1[t]))

        print ("\n\033[1mDonor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_2[t]),
                                  np.asarray(Y_pred_2[t]))

        print ("\n\033[1mTraining set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_train[:len(idx_valid)]:

            X = input_f['X' + str(idx)][:]
            Y = target_f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):

                is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

        print ("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_1[t]),
                                  np.asarray(Y_pred_1[t]))

        print ("\n\033[1mDonor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_2[t]),
                                  np.asarray(Y_pred_2[t]))

        print ("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
        print ("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        print ("--------------------------------------------------------------")

        model.save('./Models/RobertaAI_c400.h5')

        if (epoch_num+1) >= 6*len(idx_train):
            kb.set_value(model_m.optimizer.lr,
                         0.5*kb.get_value(model_m.optimizer.lr))
            # Learning rate decay

input_f.close()
target_f.close()
        
###############################################################################