
MODEL_NAME = "bert-base-multilingual-cased"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "./results"
DATA_DIR = "./data"
FINAL_MODEL_DIR = "./results/final_model"

NUM_TRAIN_SAMPLES = 3000

LABELS = [
    "O",
    "B-BRAND",     "I-BRAND",
    "B-MODEL",     "I-MODEL",
    "B-PRICE",     "I-PRICE",
    "B-CONDITION", "I-CONDITION",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

ENTITY_COLORS = {
    "BRAND":     "\033[94m",
    "MODEL":     "\033[92m",
    "PRICE":     "\033[93m",
    "CONDITION": "\033[95m",
}
RESET_COLOR = "\033[0m"
