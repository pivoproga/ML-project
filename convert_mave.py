
import json
import os
import random
from typing import List, Dict, Tuple

try:
    from datasets import load_dataset
except ImportError as _err:
    raise ImportError("Установите: pip install datasets") from _err

WIKIANN_TO_LABEL = {
    3: "BRAND",
    4: "BRAND",
}

def convert_wikiann_record(record: Dict) -> Dict:
    tokens = record["tokens"]
    wikiann_tags = record["ner_tags"]

    labels = []
    prev_brand = False
    for tag in wikiann_tags:
        if tag == 3:
            labels.append("B-BRAND")
            prev_brand = True
        elif tag == 4 and prev_brand:
            labels.append("I-BRAND")
        else:
            labels.append("O")
            prev_brand = (tag == 4)

    text = " ".join(tokens)
    entities = _labels_to_entities(tokens, labels, text)

    return {
        "text":     text,
        "tokens":   tokens,
        "labels":   labels,
        "entities": entities,
        "source":   "wikiann",
    }

def _labels_to_entities(tokens: List[str], labels: List[str], text: str) -> List[Dict]:
    entities = []
    i = 0

    tok_starts = []
    pos = 0
    for tok in tokens:
        idx = text.find(tok, pos)
        tok_starts.append(idx if idx >= 0 else pos)
        pos = (idx if idx >= 0 else pos) + len(tok)

    while i < len(labels):
        if labels[i].startswith("B-"):
            ent_type = labels[i][2:]
            start    = tok_starts[i]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{ent_type}":
                j += 1
            end = tok_starts[j - 1] + len(tokens[j - 1])
            entities.append({
                "label": ent_type,
                "start": start,
                "end":   end,
                "value": text[start:end],
            })
            i = j
        else:
            i += 1
    return entities

def has_brand(record: Dict) -> bool:
    return any(t in (3, 4) for t in record["ner_tags"])

def load_wikiann(
    lang: str,
    max_samples: int,
    seed: int = 42,
) -> List[Dict]:
    print(f"  Загружаем wikiann/{lang}...")
    ds = load_dataset("wikiann", lang, split="train")
    print(f"  Всего примеров: {len(ds)}")

    with_org = [i for i, r in enumerate(ds) if has_brand(r)]
    print(f"  С ORG-сущностью: {len(with_org)}")

    random.seed(seed)
    chosen = random.sample(with_org, min(max_samples, len(with_org)))

    converted = []
    for idx in chosen:
        rec = convert_wikiann_record(ds[idx])
        if rec["tokens"]:
            converted.append(rec)

    print(f"  Конвертировано: {len(converted)}")
    return converted

def load_and_convert_mave(
    max_samples: int = 5000,
    seed: int = 42,
    langs: Tuple[str, ...] = ("en", "ru"),
) -> List[Dict]:
    per_lang = max_samples // len(langs)
    all_data: List[Dict] = []

    for lang in langs:
        chunk = load_wikiann(lang, per_lang, seed)
        all_data.extend(chunk)

    random.seed(seed)
    random.shuffle(all_data)
    return all_data[:max_samples]

def merge_with_synthetic(
    wikiann_data: List[Dict],
    synthetic_dir: str = "data",
    output_dir: str = "data",
    real_fraction: float = 0.35,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        path = os.path.join(synthetic_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"  [!] Не найден {path} — пропускаем")
            continue

        with open(path, encoding="utf-8") as f:
            synthetic: List[Dict] = json.load(f)

        n_real = int(len(synthetic) * real_fraction / (1 - real_fraction))
        chunk  = wikiann_data[:n_real] if split == "train" else wikiann_data[n_real: n_real + n_real // 4]

        combined = synthetic + chunk
        random.shuffle(combined)

        out_path = os.path.join(output_dir, f"{split}_combined.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        print(f"  {split:5s}: synthetic={len(synthetic):4d} + wikiann={len(chunk):4d} = {len(combined):4d}  →  {out_path}")

def print_stats(data: List[Dict], name: str = "dataset") -> None:
    from collections import Counter
    lbl_cnt: Counter = Counter()
    for rec in data:
        for lbl in rec["labels"]:
            if lbl != "O":
                lbl_cnt[lbl] += 1

    print(f"\n── {name} ({len(data)} примеров) ──")
    for lbl, cnt in sorted(lbl_cnt.items()):
        print(f"  {lbl:<20} {cnt:>6}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Конвертация WikiANN в NER-формат маркетплейса")
    parser.add_argument("--lang",        default="both", choices=["en", "ru", "both"],
                        help="Язык WikiANN (default: both)")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Максимум примеров (default: 5000)")
    parser.add_argument("--no-merge",    action="store_true",
                        help="Не смешивать с синтетикой")
    args = parser.parse_args()

    langs = ("en", "ru") if args.lang == "both" else (args.lang,)

    print(f"\nКонвертация WikiANN ({', '.join(langs)})...")
    data = load_and_convert_mave(max_samples=args.max_samples, langs=langs)

    os.makedirs("data", exist_ok=True)
    out = "data/wikiann_converted.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {out} ({len(data)} примеров)")

    print_stats(data, "WikiANN converted")

    if not args.no_merge:
        print("\nСмешивание с синтетическим датасетом...")
        merge_with_synthetic(data)
        print("\nДля обучения на смешанном датасете:")
        print("  python main.py --train --use-combined")
