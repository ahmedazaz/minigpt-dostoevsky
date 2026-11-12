from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------------- Paths ----------------
DATA_DIR = BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# ---------------- Data split (applied to EACH novel) ----------------
TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.07  # test = remaining 0.03

# ---------------- Model (character-level) ----------------
EMBEDDING_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
CONTEXT_LENGTH = 256
DROPOUT = 0.1

# ---------------- Training ----------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 100
EVAL_BATCHES = 50
SEED = 42

# ---------------- Chinchilla budget ----------------
# Hoffmann et al. 2022: compute-optimal training uses ~20 tokens per parameter.
# Muennighoff et al. 2023: repeating data up to ~4 epochs is almost as good as
# fresh data, so we never repeat the training set more than MAX_EPOCHS times.
TOKENS_PER_PARAM = 20
MAX_EPOCHS = 4
