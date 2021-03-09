from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import utils.constants as constants
from networks.play_lmp import PlayLMPNet
from networks.play_lmp_states import PlayLMPStatesNet
from data_module import PlayLMPSimStatesDataModule


def main():
    # trainer options
    default_root_dir=constants.CHECKPOINT_PATH
    gpus = constants.GPUS
    num_nodes = constants.NUM_NODES
    accelerator = constants.ACCELERATOR
    plugins = constants.PLUGINS
    precision = constants.PRECISION
    max_steps = constants.MAX_STEPS
    limit_train_batches = constants.LIMIT_TRAIN_BATCHES
    val_check_interval = constants.VAL_CHECK_INTERVAL
    stochastic_weight_avg=constants.SWA
    fast_dev_run = constants.FAST_DEV_RUN
    log_gpu_memory=constants.LOG_GPU_MEMORY
    profiler=constants.PROFILER

    # hyperparams
    epochs = constants.N_EPOCH
    window_size = constants.WINDOW_SIZE
    out_features_visual = constants.OUT_FEATURES_VISUAL
    out_features_action = constants.OUT_FEATURES_ACTION
    bs = constants.BS
    lr = constants.LR
    beta = constants.BETA
    num_mix = constants.NUM_MIX
    num_workers = constants.NUM_WORKERS

    # logging
    '''
    ex usage: tensorboard --logdir ./lightning_logs
    '''
    logger = TensorBoardLogger('tb_logs', name='my_model')

    # init model
    model = PlayLMPStatesNet(lr=lr, beta=beta, num_mix=num_mix)

    # init data
    data_module = PlayLMPSimStatesDataModule(
        constants.DATA_ROOT_PATH.as_posix(),
        seq_len=window_size,
        O_features=out_features_visual,
        a_features=out_features_action,
        bs=bs,
        num_workers=num_workers,
    )

    # train
    '''
    Checkpoint loading for infernce:
        model = MyLightingModule.load_from_checkpoint(PATH)
        model.eval()
        y_hat = model(x)

    Restore training states:
        model = MyLightingModule()
        trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
        # automatically restores model, epoch, step, LR schedulers, apex, etc...
        trainer.fit(model)

    Other optimization tricks to consider:
        Auto-scaling batch size
    '''
    trainer = pl.Trainer(
        default_root_dir=default_root_dir
        gpus=gpus,
        num_nodes=num_nodes,
        accelerator=accelerator,
        plugins=plugins
        precision=precision,
        val_check_interval=val_check_interval,
        fast_dev_run=fast_dev_run,
        logger=logger,
        log_gpu_memory=log_gpu_memory,
        profiler=profiler,
    )
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()