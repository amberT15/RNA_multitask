import wandb
wandb.login()
import tensorflow as tf
import sys
import pandas as pd
import mpra_model
from gopher import utils
from ray import air,tune
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.wandb import WandbLoggerCallback

cell_type = 'HepG2'
RESULTS_DIR = '/home/ztang/multitask_RNA/evaluation/lenti_MPRA/results/'
data_dir = '/home/ztang/multitask_RNA/data/lenti_MPRA_embed/HepG2_2B5_1000G/'

config = {
    'input_shape': (41,2560),
    'reduce_dim':60,
    'conv1_filter': 119,
    'conv1_kernel':8,
    'activation':'exponential',
    'dropout1':0.2,
    'res_filter':3,
    'res_layers':4,
    'res_pool':4,
    'res_dropout':0.2,
    'conv2_filter':115,
    'conv2_kernel':7,
    'pool2_size':2,
    'dropout2':0.2,
    'dense':128,
    'dense2':128,
    'l_rate':0.001,
    'batch_size':256
}


def representation_training(config):
    trainset = mpra_model.make_dataset(data_dir, 'train', mpra_model.load_stats(data_dir),
                            batch_size=config['batch_size'])
    validset = mpra_model.make_dataset(data_dir, 'valid', mpra_model.load_stats(data_dir),
                            batch_size=config['batch_size'])
    model = mpra_model.rep_cnn(config['input_shape'],config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['l_rate'])
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['mse'])
    model.summary()
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-8)

    model.fit(
         trainset,
          epochs=100,
          batch_size=config['batch_size'],
          shuffle=False,
          validation_data=validset,
          callbacks=[earlyStopping_callback,reduce_lr,
                     TuneReportCallback({"loss": "loss","val_loss":'val_loss'})]
      )

def tune_lentiMPRA(num_training_iterations,num_samples):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=100, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(representation_training, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg = HyperOptSearch(),
            metric="val_loss",
            mode="min",
            scheduler=sched,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="lentiMPRA_rep_training",
            stop={"training_iteration": num_training_iterations},
            callbacks=[WandbLoggerCallback(project="nuc_transfromer_lentiMPRA")]
        ),
        param_space=config,
    )
    results = tuner.fit()
    result_df = results.get_dataframe()
    result_df.to_csv(RESULTS_DIR + 'tune_all.csv')
    best_df = results.get_best_result()
    best_df.to_csv(RESULTS_DIR+'best_result.csv')
    # print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == "__main__":
    
    tune_lentiMPRA(num_training_iterations=100,num_samples=50)
    

