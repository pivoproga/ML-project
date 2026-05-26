
import json
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

@dataclass
class Sample:
    text: str
    entities: List[Entity] = field(default_factory=list)

BRANDS = [
    "Apple", "Samsung", "Sony", "LG", "Xiaomi", "Huawei", "Nokia", "Asus",
    "Lenovo", "HP", "Dell", "Acer", "Canon", "Nikon", "Bosch", "Philips",
    "Dyson", "Panasonic", "JBL", "Bose", "MSI", "Logitech", "Razer",
    "Adidas", "Nike", "Puma", "Reebok", "New Balance", "Corsair",
    "Kingston", "WD", "Seagate", "Transcend",
    "iPhone", "iPad", "MacBook", "AirPods",
    "Galaxy", "Redmi", "Poco",
    "PlayStation", "Xbox",
    "Самсунг", "Сони", "Сяоми", "Хуавей", "Нокиа", "Асус",
    "Леново", "Бош", "Филипс", "Панасоник", "Найк", "Адидас",
    "Эпл", "Дайсон", "Кэнон", "Никон", "Рибок",
    "Айфон", "Плейстейшн",
]

MODELS_BY_BRAND: Dict[str, List[str]] = {
    "Самсунг":  ["Galaxy S21 Ultra", "Galaxy S22+", "Galaxy A53", "Galaxy A32", "Galaxy Z Fold 3"],
    "Сони":     ["PlayStation 5", "WH-1000XM5", "Xperia 1 IV", "WF-1000XM4"],
    "Сяоми":    ["Redmi Note 11 Pro", "Mi 12 Pro", "Poco X4 GT", "Mi Band 7", "Poco F4"],
    "Хуавей":   ["P50 Pro", "Nova 9", "MatePad 11", "Watch GT 3"],
    "Нокиа":    ["G50", "X30", "5.4", "3310"],
    "Асус":     ["ROG Phone 6", "ZenBook 14", "TUF Gaming F15"],
    "Леново":   ["ThinkPad X1 Carbon", "IdeaPad 5 Pro", "Legion 5 Pro"],
    "Бош":      ["GSB 18V-55", "GBH 2-28 F", "GWS 18V-10"],
    "Филипс":   ["Series 9000", "5000", "HX9352"],
    "Панасоник":["Lumix G9", "TX2", "HC-V800"],
    "Найк":     ["Air Max 90", "React Infinity", "Pegasus 39", "Zoom Fly 4"],
    "Адидас":   ["Ultraboost 22", "Stan Smith", "NMD R1", "Gazelle"],
    "Эпл":      ["iPhone 13", "iPhone 12 Pro", "MacBook Air M1", "AirPods Pro"],
    "Дайсон":   ["V15 Detect", "Airwrap Complete", "V8 Animal"],
    "Кэнон":    ["EOS R6 Mark II", "EOS 90D", "EOS M50"],
    "Никон":    ["Z6 II", "D7500", "Z 50"],
    "Рибок":    ["Classic Leather", "Nano X2", "Floatride Energy 4"],
    "iPhone":   ["14 Pro 256gb", "14 Pro 128gb", "14 256gb", "14 Plus 256gb",
                 "13 Pro Max 512gb", "13 Pro 256gb", "13 128gb", "13 64gb",
                 "12 Pro 256gb", "12 64gb", "12 Pro Max 256gb",
                 "11 128gb", "11 Pro 256gb", "SE 64gb", "XR 128gb"],
    "Айфон":    ["14 Про 256гб", "14 Про 128гб", "13 Про 256гб",
                 "13 128гб", "12 64гб", "11 128гб"],
    "iPad":     ["Pro 12.9 256gb", "Pro 11 128gb", "Air 5 256gb",
                 "mini 6 64gb", "10 256gb", "9 64gb"],
    "MacBook":  ["Air M1 8gb 256gb", "Air M1 8gb 512gb",
                 "Pro 14 16gb 512gb", "Pro 13 8gb 256gb", "Air M2 8gb 256gb"],
    "AirPods":  ["Pro 2", "Pro", "3", "2", "Max"],
    "Galaxy":   ["S22 Ultra 256gb", "S22+ 128gb", "S22 256gb",
                 "S21 256gb", "A53 128gb", "A32 64gb",
                 "Z Fold 3 256gb", "Tab S8 256gb"],
    "Redmi":    ["Note 11 Pro 128gb", "Note 11 64gb", "Note 10 Pro 128gb",
                 "9A 32gb", "10C 64gb", "Note 12 256gb"],
    "Poco":     ["X4 GT 128gb", "F4 256gb", "M4 Pro 64gb", "X5 Pro 256gb"],
    "PlayStation": ["5", "5 Digital Edition", "4 Pro 1TB", "4 500gb"],
    "Xbox":     ["Series X", "Series S", "One X 1TB"],
    "Плейстейшн": ["5", "4 Про", "4 1тб"],
    "Apple":    ["iPhone 13", "iPhone 12 Pro Max", "iPhone 11", "iPhone SE",
                 "MacBook Air M1", "MacBook Pro 14", "iPad Pro 12.9",
                 "AirPods Pro", "Apple Watch Series 7", "iMac 24"],
    "Samsung":  ["Galaxy S21 Ultra", "Galaxy S22+", "Galaxy A53 5G",
                 "Galaxy Tab S8", "Galaxy Watch 5", "Galaxy Buds Pro",
                 "Galaxy Z Fold 3", "Galaxy A32"],
    "Sony":     ["PlayStation 5", "WH-1000XM5", "A7 IV", "Xperia 1 IV",
                 "WF-1000XM4", "SRS-XB43", "ZV-E10"],
    "Xiaomi":   ["Redmi Note 11 Pro", "Mi 12 Pro", "Poco X4 GT", "Mi Band 7",
                 "Redmi Watch 2", "Mi Pad 5 Pro", "Poco F4"],
    "LG":       ["OLED C2 55", "Gram 16", "K92 5G", "UltraFine 27"],
    "Asus":     ["ROG Phone 6", "ZenBook 14", "TUF Gaming F15", "ROG Strix G15"],
    "Lenovo":   ["ThinkPad X1 Carbon", "IdeaPad 5 Pro", "Legion 5 Pro",
                 "Tab P12 Pro", "ThinkPad E14"],
    "HP":       ["Spectre x360 14", "EliteBook 840 G9", "Pavilion 15",
                 "OMEN 16", "Envy 13"],
    "Dell":     ["XPS 13 Plus", "Inspiron 15 3000", "Alienware m15 R7",
                 "Latitude 7420", "G15 Gaming"],
    "Bosch":    ["GSB 18V-55", "GBH 2-28 F", "GAS 18V-10", "PST 700 E",
                 "GWS 18V-10"],
    "Dyson":    ["V15 Detect", "Airwrap Complete", "Hot+Cool HP07",
                 "Purifier Cool TP07", "V8 Animal"],
    "Logitech": ["MX Master 3", "G Pro X Superlight", "MX Keys Mini",
                 "G915 TKL", "C920 HD Pro"],
    "Razer":    ["Blade 15", "DeathAdder V3", "BlackShark V2 Pro",
                 "Huntsman Mini", "Kishi V2"],
    "MSI":      ["Katana GF66", "Raider GE76", "Creator Z16", "GP76 Leopard"],
    "Canon":    ["EOS R6 Mark II", "EOS 90D", "RF 50mm f/1.8", "EOS M50"],
    "Nikon":    ["Z6 II", "D7500", "Z 50", "NIKKOR Z 24-70mm"],
    "Kingston": ["HyperX Fury 16GB", "DataTraveler 256GB", "KC3000 1TB"],
    "WD":       ["My Passport 2TB", "Black SN850X 1TB", "Blue 4TB HDD"],
    "Seagate":  ["Barracuda 2TB", "FireCuda 530 2TB", "Expansion 5TB"],
}

DEFAULT_MODELS = [
    "Pro", "Ultra", "Plus", "Lite", "Max", "Neo", "X2", "Air",
    "GT Pro", "Special Edition", "Elite", "Smart",
]

CONDITIONS_RU = [
    "новый", "новое", "новая",
    "б/у", "б/у (хорошее состояние)", "б/у (отличное состояние)",
    "хорошее состояние", "отличное состояние", "идеальное состояние",
    "хорошем состоянии", "отличном состоянии",
    "как новый", "как новое", "почти новый",
    "без дефектов", "без царапин", "без следов использования",
    "запечатанный", "в упаковке", "нерабочий", "требует ремонта",
    "рабочий", "исправный",
    "бу", "б у", "б.у.", "пользованный",
    "10/10", "9/10", "8/10", "5/5", "4/5",
    "отл", "хор", "норм", "норм сост",
    "идеал", "идеальн", "почти идеал",
    "следов нет", "царапин нет", "не бит не крашен",
    "сост 5+", "состояние отл", "сост хор",
    "состояние на фото", "сост на фото", "см фото",
    "состояние смотрите на фото", "на фото всё видно",
    "фото = состояние",
    "акб 100%", "акб 98%", "акб 95%", "акб 90%", "акб 85%",
    "батарея 100%", "батарея 98%", "акб в норме",
    "все работает", "всё работает", "работает отлично",
    "без нюансов", "нюансов нет",
    "оригинал", "не рефка", "не восстановленный",
    "полный комплект", "коробка есть", "документы есть",
]

CONDITIONS_EN = [
    "new", "brand new", "sealed", "unopened",
    "used", "like new", "good condition", "excellent condition",
    "perfect condition", "mint condition",
    "fair condition", "for parts", "refurbished",
]

CONDITIONS = CONDITIONS_RU + CONDITIONS_EN

CURRENCIES_RU = ["руб.", "₽", "рублей", "р.", "RUB", "руб"]
CURRENCIES_EN = ["$", "USD", "€", "EUR"]
CURRENCIES = CURRENCIES_RU + CURRENCIES_EN

TEMPLATES_RU = [
    "Продаю {brand} {model}. Состояние: {condition}. Цена: {price} {currency}.",
    "{brand} {model}, состояние — {condition}, {price} {currency}. Торг уместен.",
    "Продается {brand} {model} в {condition} состоянии за {price} {currency}.",
    "Срочно! {brand} {model} — {price} {currency}. Состояние: {condition}.",
    "Отдам {brand} {model} ({condition}) за {price} {currency}.",
    "{brand} {model}. Цена {price} {currency}. Состояние {condition}.",
    "Продам {brand} {model}, {condition}, цена {price} {currency}.",
    "Предлагаю {brand} {model} за {price} {currency}. {condition}.",
    "ПРОДАМ {brand} {model}! {condition}! {price} {currency}!",
    "Телефон {brand} {model} в состоянии {condition}. {price} {currency}.",
    "Ноутбук {brand} {model}, {condition}, за {price} {currency}.",
    "{brand} {model} — отдам за {price} {currency}. Состояние: {condition}.",
    "Хочу продать {brand} {model}. Состояние {condition}. Прошу {price} {currency}.",
    "Продаю {brand} {model}. {condition}. Причина: апгрейд. {price} {currency}.",
    "Объявление: {brand} {model}, {price} {currency}, {condition}.",
    "Лот: {brand} {model}. Состояние — {condition}. Стартовая цена {price} {currency}.",
    "{brand} {model} отдам за {price} {currency}, сост {condition}",
    "проадю {brand} {model} {condition} {price} {currency} торг",
    "{brand} {model} {price}{currency} {condition} звоните",
    "Есть {brand} {model}. {condition}. Цена договорная, примерно {price} {currency}.",
    "{brand} {model} за {price} {currency}! Сост: {condition}, всё работает.",
    "Избавляюсь от {brand} {model} ({condition}), {price} {currency}.",
    "СРОЧНО {brand} {model} {price} {currency} {condition}!!!",
    "{brand} {model}, пользовался {condition}, отдам за {price} {currency}.",
    "Продам {brand} {model}. {condition}. Цена {price} {currency} торг.",
    "Продаю {brand} {model}, {condition}. Звонить по номеру. {price} {currency}.",
    "{brand} {model}. {condition}, все работает. Цена {price} {currency}, торг уместен.",
    "Продам {brand} {model}, все работает, {condition}. {price} {currency}.",
    "Отдам {brand} {model}. {condition}. Цена {price} {currency}, обмен не интересует.",
    "{brand} {model} {condition} цена {price} {currency} торг звоните",
    "Продам {brand} {model}, {condition}. Комплект полный. {price} {currency}.",
    "{brand} {model}. {condition}. Срочно! Цена {price} {currency}.",
    "Б/у {brand} {model}, {condition}. Отдам за {price} {currency}.",
    "Продаю {brand} {model} {condition}. {price} {currency} последняя цена.",
]

TEMPLATES_EN = [
    "Selling {brand} {model}. Condition: {condition}. Price: {price} {currency}.",
    "{brand} {model}, {condition}, asking {price} {currency}. Negotiable.",
    "For sale: {brand} {model} in {condition} condition. {price} {currency}.",
    "Urgent! {brand} {model} — {price} {currency}. {condition}.",
    "{brand} {model} ({condition}) for {price} {currency}.",
    "I'm selling my {brand} {model}, {condition}, price {price} {currency}.",
    "SELLING {brand} {model}! {condition}! {price} {currency}!",
    "Listing: {brand} {model}. {condition}. Asking {price} {currency}.",
    "Item: {brand} {model}, {condition}, {price} {currency} OBO.",
]

TEMPLATES = TEMPLATES_RU + TEMPLATES_EN

def get_model_for_brand(brand: str) -> str:
    if brand in MODELS_BY_BRAND:
        return random.choice(MODELS_BY_BRAND[brand])
    return random.choice(DEFAULT_MODELS)

def generate_price() -> str:
    price = random.randint(500, 300_000)
    if price > 10_000:
        price = round(price, -3)
    elif price > 1_000:
        price = round(price, -2)
    if price >= 1000 and random.random() < 0.15:
        k = price // 1000
        suffix = random.choice(['к', 'К', 'k', 'K'])
        return f"{k}{suffix}"
    return str(price)

def find_span(text: str, substr: str, start: int = 0) -> Tuple[int, int]:
    idx = text.find(substr, start)
    if idx == -1:
        return -1, -1
    return idx, idx + len(substr)

def remove_overlaps(entities: List[Entity]) -> List[Entity]:
    entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    result: List[Entity] = []
    last_end = -1
    for e in entities:
        if e.start >= last_end:
            result.append(e)
            last_end = e.end
    return result

def generate_sample() -> Optional[Sample]:
    brand = random.choice(BRANDS)
    model = get_model_for_brand(brand)
    condition = random.choice(CONDITIONS)
    price_val = generate_price()
    currency = random.choice(CURRENCIES)

    template = random.choice(TEMPLATES)
    text = template.format(
        brand=brand, model=model,
        condition=condition, price=price_val, currency=currency,
    )

    entities: List[Entity] = []

    b_start, b_end = find_span(text, brand)
    if b_start >= 0:
        entities.append(Entity(brand, "BRAND", b_start, b_end))

    search_from = b_end if b_end >= 0 else 0
    m_start, m_end = find_span(text, model, search_from)
    if m_start >= 0:
        entities.append(Entity(model, "MODEL", m_start, m_end))

    p_start, p_end = find_span(text, price_val)
    if p_start >= 0:
        after = text[p_end:]
        stripped = after.lstrip(" ")
        gap = len(after) - len(stripped)
        if stripped.startswith(currency):
            p_end = p_end + gap + len(currency)
        entities.append(Entity(text[p_start:p_end], "PRICE", p_start, p_end))

    c_start, c_end = find_span(text, condition)
    if c_start >= 0:
        entities.append(Entity(condition, "CONDITION", c_start, c_end))

    entities = remove_overlaps(entities)
    entities.sort(key=lambda e: e.start)

    return Sample(text, entities)

def sample_to_bio(sample: Sample) -> Tuple[List[str], List[str]]:
    text = sample.text
    tokens, tok_starts, tok_ends = [], [], []

    i = 0
    while i < len(text):
        while i < len(text) and text[i] == " ":
            i += 1
        if i >= len(text):
            break
        j = i
        while j < len(text) and text[j] != " ":
            j += 1
        tokens.append(text[i:j])
        tok_starts.append(i)
        tok_ends.append(j)
        i = j

    labels = ["O"] * len(tokens)

    for ent in sample.entities:
        first = True
        for k, (ts, te) in enumerate(zip(tok_starts, tok_ends)):
            if te <= ent.start or ts >= ent.end:
                continue
            labels[k] = f"B-{ent.label}" if first else f"I-{ent.label}"
            first = False

    return tokens, labels

def generate_dataset(n: int = 3000, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    dataset = []
    attempts = 0
    while len(dataset) < n and attempts < n * 3:
        attempts += 1
        s = generate_sample()
        if s is None:
            continue
        tokens, labels = sample_to_bio(s)
        if not tokens:
            continue
        dataset.append({
            "text":    s.text,
            "tokens":  tokens,
            "labels":  labels,
            "entities": [
                {"text": e.text, "label": e.label,
                 "start": e.start, "end": e.end}
                for e in s.entities
            ],
        })
    return dataset

def split_dataset(
    data: List[Dict],
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    t = int(n * train)
    v = int(n * (train + val))
    return data[:t], data[t:v], data[v:]

if __name__ == "__main__":
    import os

    print("Генерация датасета...")
    ds = generate_dataset(3000)
    print(f"Сгенерировано {len(ds)} примеров")

    for s in ds[:3]:
        print("\n─" * 60)
        print("Текст:  ", s["text"])
        print("Токены: ", s["tokens"])
        print("Метки:  ", s["labels"])

    tr, va, te = split_dataset(ds)
    print(f"\nРазбивка: train={len(tr)}, val={len(va)}, test={len(te)}")

    os.makedirs("data", exist_ok=True)
    for name, split in [("train", tr), ("val", va), ("test", te)]:
        with open(f"data/{name}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
    print("Данные сохранены в data/")
