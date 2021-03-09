from pathlib import Path

# Data constants
DATA_ROOT_PATH = Path.cwd().parents[1] / "datasets"

# Checkpoint route
CHECKPOINT_PATH = Path.cwd() / "checkpoints"

# Trainer flags
GPUS = 3
NUM_NODES = 1
ACCELERATOR = "ddp"
PLUGINS = "ddp_sharded"
PRECISION = 16
MAX_STEPS = 10
LIMIT_TRAIN_BATCHES = 10
VAL_CHECK_INTERVAL = 0.5  # i.e. 0.1 means 10 times every epoch
SWA = True
FAST_DEV_RUN = False
LOG_GPU_MEMORY = True
PROFILER = "simple"  # one of 'simple' or 'advanced' (i.e. function level) or 'pytorch' (https://pytorch-lightning.readthedocs.io/en/stable/profiler.html)

# Network constants
OUT_FEATURES_VISUAL = 56  # for image input: 64
OUT_FEATURES_ACTION = 8
OUT_FEATURES_PLAN = 256
NUM_MIX = 10  # number of mixtures for discretized logistic mixture

# Training hyperparameters
N_EPOCH = 50
WINDOW_SIZE = 32
BS = 64
LR = 2e-4
BETA = 0.01
NUM_WORKERS = 12
