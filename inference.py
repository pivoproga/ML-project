
import os
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config import ENTITY_COLORS, RESET_COLOR

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def normalize_text(text: str) -> str:
    import re
    text = re.sub(r',([^\s])', r', \1', text)
    text = re.sub(r'([\.!?;:])([^\s\d\.])', r'\1 \2', text)
    text = re.sub(r'(\d+)([а-яёa-z₽])', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'([^\s])\(', r'\1 (', text)
    text = re.sub(r'\)([^\s,\.!?])', r') \1', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    return text

def predict(text: str, tokenizer, model, device=None) -> List[Dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = normalize_text(text)

    words = text.split()
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    word_ids = enc.word_ids()

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits

    pred_ids = logits.argmax(-1).squeeze().tolist()
    id2label = model.config.id2label

    word_labels: List[str] = []
    prev_wid = None
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            word_labels.append(id2label[pid])
        prev_wid = wid

    entities = bio_to_spans(words, word_labels, text)
    return entities

def bio_to_spans(words: List[str], labels: List[str], original_text: str) -> List[Dict]:
    entities = []
    current_ent = None
    current_words = []

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            if current_ent:
                entities.append(_build_entity(current_words, current_ent, original_text))
            current_ent = label[2:]
            current_words = [word]
        elif label.startswith("I-") and current_ent == label[2:]:
            current_words.append(word)
        else:
            if current_ent:
                entities.append(_build_entity(current_words, current_ent, original_text))
            current_ent = None
            current_words = []

    if current_ent:
        entities.append(_build_entity(current_words, current_ent, original_text))

    return entities

def _build_entity(words: List[str], label: str, original_text: str) -> Dict:
    span_text = " ".join(words)
    start = original_text.find(span_text)
    end   = start + len(span_text) if start >= 0 else -1
    return {"text": span_text, "label": label, "start": start, "end": end}

def pretty_print(text: str, entities: List[Dict]) -> None:
    print("\n" + "─" * 70)
    print(f"ТЕКСТ: {text}")
    print("─" * 70)
    for ent in entities:
        color = ENTITY_COLORS.get(ent["label"], "")
        print(f"  {color}[{ent['label']}]{RESET_COLOR}  {ent['text']!r}")
    if not entities:
        print("  (сущности не найдены)")
    print()

DEMO_TEXTS = [
    "Продам iPhone 14 Pro 256gb, состояние отличное, акб 98%. Цена 65000 руб. Торг.",

    "Самсунг Galaxy S22 Ultra 256гб, бу, всё работает, царапин нет. 45к руб торг",

    "Продаю Redmi Note 11 Pro 128gb. Состояние на фото. Акб 91%. 15000 р.",

    "Lenovo ThinkPad E14, 10/10, без дефектов, полный комплект. Цена 42000 ₽.",

    "prodam iPhone 13 128gb, sostoyanie kak noviy, tsena 48000 rub",

    "Продаётся PlayStation 5 Digital Edition, как новое, полный комплект. Торг уместен.",

    "Selling Samsung Galaxy Z Fold 3 256gb. Excellent condition, no scratches. $650 OBO.",

    "Galaxy A53 128gb бу норм сост 12к",

    "Sony WH-1000XM5 б/у (отличное состояние), коробка есть. 18000 руб.",

    "СРОЧНО! ПРОДАМ ASUS ROG Phone 6 256GB! СОСТОЯНИЕ ИДЕАЛ! ЦЕНА 55000 РУБ!!!",

    "MacBook Air M1 8gb 256gb, как новый, все работает. 75 000 ₽.",

    "AirPods Pro 2 новые запечатанные 15000р",
]

def run_inference(model_dir: str = "./results/final_model") -> None:
    print("\n" + "=" * 70)
    print("     ДЕМОНСТРАЦИЯ: ИЗВЛЕЧЕНИЕ СУЩНОСТЕЙ ИЗ ОБЪЯВЛЕНИЙ")
    print("=" * 70)

    tokenizer, model = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for text in DEMO_TEXTS:
        entities = predict(text, tokenizer, model, device)
        pretty_print(text, entities)

if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./results/final_model"
    run_inference(model_dir)
