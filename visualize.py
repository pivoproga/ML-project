
import json
import os
import re
from collections import Counter
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = {
    "train":     "#4C72B0",
    "val":       "#DD8452",
    "f1":        "#55A868",
    "precision": "#4C72B0",
    "recall":    "#C44E52",
    "BRAND":     "#4C72B0",
    "MODEL":     "#55A868",
    "PRICE":     "#C44E52",
    "CONDITION": "#8172B2",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        120,
})

def parse_report(report: str) -> Dict[str, Dict[str, float]]:
    entity_re = re.compile(
        r"^\s+(BRAND|MODEL|PRICE|CONDITION)\s+"
        r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)",
        re.MULTILINE,
    )
    result = {}
    for m in entity_re.finditer(report):
        label = m.group(1)
        result[label] = {
            "precision": float(m.group(2)),
            "recall":    float(m.group(3)),
            "f1":        float(m.group(4)),
            "support":   int(m.group(5)),
        }
    return result

def _plot_loss(ax, history: Dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "o-", color=COLORS["train"],
            label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "o-", color=COLORS["val"],
            label="Val Loss", linewidth=2)
    ax.set_title("Функция потерь", fontweight="bold")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_xticks(list(epochs))

def _plot_metrics(ax, history: Dict) -> None:
    epochs = range(1, len(history["val_f1"]) + 1)
    ax.plot(epochs, history["val_f1"],        "o-",  color=COLORS["f1"],
            label="F1",        linewidth=2)
    ax.plot(epochs, history["val_precision"], "s--", color=COLORS["precision"],
            label="Precision", linewidth=1.6, alpha=0.85)
    ax.plot(epochs, history["val_recall"],    "^--", color=COLORS["recall"],
            label="Recall",    linewidth=1.6, alpha=0.85)

    min_v = min(min(history["val_f1"]),
                min(history["val_precision"]),
                min(history["val_recall"]))
    y_lo = max(0.0, min_v - 0.005)
    ax.set_ylim(y_lo, 1.005)
    ax.set_title("Метрики на val-выборке (zoom)", fontweight="bold")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.set_xticks(list(epochs))

    for key, vals in [("val_f1", history["val_f1"]),
                      ("val_precision", history["val_precision"]),
                      ("val_recall", history["val_recall"])]:
        last = vals[-1]
        ax.annotate(f"{last:.4f}", xy=(len(vals), last),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, va="center")

def _plot_per_entity(ax, entity_metrics: Dict) -> None:
    if not entity_metrics:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                transform=ax.transAxes)
        return

    entities   = list(entity_metrics.keys())
    precisions = [entity_metrics[e]["precision"] for e in entities]
    recalls    = [entity_metrics[e]["recall"]    for e in entities]
    f1s        = [entity_metrics[e]["f1"]        for e in entities]

    x     = np.arange(len(entities))
    width = 0.26

    b1 = ax.bar(x - width, precisions, width, label="Precision",
                color=COLORS["precision"], alpha=0.85)
    b2 = ax.bar(x,          f1s,        width, label="F1",
                color=COLORS["f1"],        alpha=0.85)
    b3 = ax.bar(x + width,  recalls,   width, label="Recall",
                color=COLORS["recall"],    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(entities, fontsize=10)
    ax.set_ylim(0.95, 1.01)
    ax.set_title("Метрики по классам на тесте", fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h >= 0.95:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

def _plot_dataset_stats(ax, data_dir: str) -> None:
    splits = ["train", "val", "test"]
    counts_by_split = {}
    entity_counts_by_split: Dict[str, Counter] = {}

    for split in splits:
        path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data: List[Dict] = json.load(f)
        counts_by_split[split] = len(data)
        ec: Counter = Counter()
        for rec in data:
            for lbl in rec["labels"]:
                if lbl.startswith("B-"):
                    ec[lbl[2:]] += 1
        entity_counts_by_split[split] = ec

    if not counts_by_split:
        ax.text(0.5, 0.5, "data/ не найден", ha="center", va="center",
                transform=ax.transAxes)
        return

    entity_types = ["BRAND", "MODEL", "PRICE", "CONDITION"]
    x     = np.arange(len(splits))
    width = 0.2
    offsets = np.linspace(-width * 1.5, width * 1.5, len(entity_types))

    for etype, offset in zip(entity_types, offsets):
        vals = [entity_counts_by_split.get(s, Counter()).get(etype, 0) for s in splits]
        ax.bar(x + offset, vals, width, label=etype,
               color=COLORS.get(etype, "#888888"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({counts_by_split.get(s, 0):,})" for s in splits])
    ax.set_title("Распределение сущностей по выборкам", fontweight="bold")
    ax.set_ylabel("Число сущностей")
    ax.legend(loc="upper right", fontsize=9)

def plot_all(history_path: str = "./results/history.json",
             results_path: str = "./results/results.json",
             data_dir:     str = "./data",
             save_dir:     str = "./results") -> None:
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    entity_metrics = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        entity_metrics = parse_report(results.get("report", ""))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "NER для торговых площадок — результаты обучения BERT\n"
        f"val F1={history['val_f1'][-1]:.4f}  |  "
        f"test F1={results.get('test_f1', 0):.4f}",
        fontsize=13, fontweight="bold",
    )

    _plot_loss(axes[0, 0], history)
    _plot_metrics(axes[0, 1], history)
    _plot_per_entity(axes[1, 0], entity_metrics)
    _plot_dataset_stats(axes[1, 1], data_dir)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(save_dir, "training_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Дашборд сохранён: {out}")
    return out

def plot_training_curves(history_path: str = "./results/history.json",
                         save_dir: str = "./results") -> None:
    plot_all(history_path=history_path, save_dir=save_dir)

def plot_entity_distribution(data_path: str = "./data/train.json",
                              save_dir: str = "./results") -> None:
    pass

def print_dataset_stats(data_dirs: Dict[str, str] = None) -> None:
    if data_dirs is None:
        data_dirs = {
            "train": "./data/train.json",
            "val":   "./data/val.json",
            "test":  "./data/test.json",
        }

    print("\n" + "=" * 54)
    print("  СТАТИСТИКА ДАТАСЕТА")
    print("=" * 54)

    for split, path in data_dirs.items():
        if not os.path.exists(path):
            print(f"\n  {split.upper()} — файл не найден")
            continue
        with open(path, encoding="utf-8") as f:
            data: List[Dict] = json.load(f)

        entity_counts: Counter = Counter()
        token_counts  = []
        for rec in data:
            for lbl in rec["labels"]:
                if lbl.startswith("B-"):
                    entity_counts[lbl[2:]] += 1
            token_counts.append(len(rec["tokens"]))

        print(f"\n  {split.upper()} ({len(data):,} примеров)")
        print(f"    Средняя длина (токенов): {sum(token_counts)/len(token_counts):.1f}")
        for etype, cnt in sorted(entity_counts.items()):
            print(f"    {etype:<12}: {cnt:,} сущностей")

    print("=" * 54)

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    print_dataset_stats()
    if os.path.exists("./results/history.json"):
        plot_all()
    else:
        print("history.json не найден — запустите train.py сначала")
