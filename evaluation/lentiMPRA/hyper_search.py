import wandb
wandb.login()
import tensorflow as tf
import sys
import pandas as pd
import mpra_model
from gopher import utils
from ray import air,tune
from ray.air import session
from pathlib import Path
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.wandb import WandbLoggerCallback

cell_type = 'K562'
RESULTS_DIR = '/home/ztang/multitask_RNA/evaluation/lenti_MPRA/results_k562/'
data_dir = '/home/ztang/multitask_RNA/data/lenti_MPRA_embed/K562_2B5_1000G/'

config = {
    'cell_type': cell_type,
    'input_shape': (41,2560),
    'reduce_dim':tune.randint(60,120),
    'conv1_filter': tune.randint(120,360),
    'conv1_kernel':tune.randint(3,8),
    'activation':'exponential',
    'dropout1':tune.choice([0.2,0.3,0.4]),
    'res_filter':tune.randint(2,4),
    'res_layers':tune.randint(1,6),
    'res_pool':tune.randint(2,5),
    'res_dropout':0.2,
    'conv2_filter':tune.randint(256,360),
    'conv2_kernel':tune.randint(2,6),
    'pool2_size':tune.randint(2,5),
    'dropout2':0.2,
    'dense':tune.randint(128,256),
    'dense2':tune.randint(64,128),
    'l_rate':tune.choice([0.001,0.0001,0.0005]),
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
    tune_trial_dir = Path(session.get_trial_dir())
    model.save_weights(f'{tune_trial_dir}/weights')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{tune_trial_dir}/weights',
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode = 'min',
                                        save_freq='epoch',)
    model.fit(
         trainset,
          epochs=100,
          batch_size=config['batch_size'],
          shuffle=False,
          validation_data=validset,
          callbacks=[earlyStopping_callback,reduce_lr,checkpoint,
                     TuneReportCallback({"loss": "loss","val_loss":'val_loss'})]
      )
    tune_trial_dir = Path(session.get_trial_dir())
    model.save_weights(f'{tune_trial_dir}/weights')

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
    

