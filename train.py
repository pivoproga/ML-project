
import json
import os
import time
import logging
from typing import List, Dict, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY,
    OUTPUT_DIR, DATA_DIR, FINAL_MODEL_DIR,
    LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
)
from dataset import MarketplaceNERDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
) -> Dict[str, float]:
    pred_seqs, true_seqs = [], []
    for pred_row, label_row in zip(preds, labels):
        pred_seq, true_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_seq.append(id2label[p])
            true_seq.append(id2label[l])
        pred_seqs.append(pred_seq)
        true_seqs.append(true_seq)

    return {
        "f1":        f1_score(true_seqs, pred_seqs),
        "precision": precision_score(true_seqs, pred_seqs),
        "recall":    recall_score(true_seqs, pred_seqs),
    }

def run_epoch(
    model,
    loader: DataLoader,
    optimizer=None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
    train: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids      = batch["labels"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
            )
            loss = out.loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

        preds  = out.logits.argmax(dim=-1).cpu().numpy()
        labels = label_ids.cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics    = compute_metrics(all_preds, all_labels, ID2LABEL)
    avg_loss   = total_loss / len(loader)

    return avg_loss, metrics

def train_model() -> Dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")

    logger.info("Загрузка данных...")
    with open(f"{DATA_DIR}/train.json", encoding="utf-8") as f:
        train_data: List[Dict] = json.load(f)
    with open(f"{DATA_DIR}/val.json", encoding="utf-8") as f:
        val_data: List[Dict] = json.load(f)

    logger.info(f"Загрузка токенизатора: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = MarketplaceNERDataset(train_data, tokenizer, LABEL2ID, MAX_LENGTH)
    val_ds   = MarketplaceNERDataset(val_data,   tokenizer, LABEL2ID, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    logger.info(f"Инициализация модели: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    logger.info(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_groups, lr=LEARNING_RATE)

    total_steps   = NUM_EPOCHS * len(train_loader)
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": [],
               "val_precision": [], "val_recall": []}
    best_f1 = 0.0

    logger.info("─" * 60)
    logger.info(f"Начало обучения | epochs={NUM_EPOCHS} | steps={total_steps}")
    logger.info("─" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_m = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        va_loss, va_m = run_epoch(model, val_loader, device=device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_m["f1"])
        history["val_precision"].append(va_m["precision"])
        history["val_recall"].append(va_m["recall"])

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={va_loss:.4f} | "
            f"val_f1={va_m['f1']:.4f} | "
            f"val_P={va_m['precision']:.4f} | "
            f"val_R={va_m['recall']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if va_m["f1"] >= best_f1:
            best_f1 = va_m["f1"]
            model.save_pretrained(FINAL_MODEL_DIR)
            tokenizer.save_pretrained(FINAL_MODEL_DIR)
            logger.info(f"  ✓ Новый лучший val F1={best_f1:.4f} — модель сохранена")

    logger.info("─" * 60)
    logger.info(f"Обучение завершено. Лучший val F1 = {best_f1:.4f}")

    with open(f"{DATA_DIR}/test.json", encoding="utf-8") as f:
        test_data: List[Dict] = json.load(f)

    test_ds     = MarketplaceNERDataset(test_data, tokenizer, LABEL2ID, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_model = AutoModelForTokenClassification.from_pretrained(FINAL_MODEL_DIR)
    best_model.to(device)

    test_loss, test_m = run_epoch(best_model, test_loader, device=device, train=False)
    logger.info(f"TEST  loss={test_loss:.4f} | F1={test_m['f1']:.4f} | "
                f"P={test_m['precision']:.4f} | R={test_m['recall']:.4f}")

    all_preds, all_labels_list = [], []
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = best_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds  = out.logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].numpy()
            all_preds.append(preds)
            all_labels_list.append(labels)

    all_preds  = np.concatenate(all_preds,       axis=0)
    all_labels_arr = np.concatenate(all_labels_list, axis=0)

    pred_seqs, true_seqs = [], []
    for p_row, l_row in zip(all_preds, all_labels_arr):
        pred_seq, true_seq = [], []
        for p, l in zip(p_row, l_row):
            if l == -100:
                continue
            pred_seq.append(ID2LABEL[p])
            true_seq.append(ID2LABEL[l])
        pred_seqs.append(pred_seq)
        true_seqs.append(true_seq)

    report = classification_report(true_seqs, pred_seqs)
    logger.info("\nClassification Report (тест):\n" + report)

    with open(f"{OUTPUT_DIR}/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    results = {**history, "best_val_f1": best_f1,
               "test_f1": test_m["f1"],
               "test_precision": test_m["precision"],
               "test_recall": test_m["recall"],
               "report": report}

    with open(f"{OUTPUT_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

if __name__ == "__main__":
    train_model()
