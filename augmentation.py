
import random
import re
from typing import List, Dict, Tuple

RU_TO_LAT: Dict[str, str] = {
    'а': 'a',  'б': 'b',  'в': 'v',  'г': 'g',  'д': 'd',
    'е': 'e',  'ё': 'yo', 'ж': 'zh', 'з': 'z',  'и': 'i',
    'й': 'j',  'к': 'k',  'л': 'l',  'м': 'm',  'н': 'n',
    'о': 'o',  'п': 'p',  'р': 'r',  'с': 's',  'т': 't',
    'у': 'u',  'ф': 'f',  'х': 'h',  'ц': 'c',  'ч': 'ch',
    'ш': 'sh', 'щ': 'sch','ъ': '',   'ы': 'y',  'ь': '',
    'э': 'e',  'ю': 'yu', 'я': 'ya',
}

BRAND_TO_CYRILLIC: Dict[str, List[str]] = {
    'Apple':    ['Эпл', 'эпл', 'ЭПЛЬ'],
    'Samsung':  ['Самсунг', 'самсунг', 'САМСУНГ'],
    'Sony':     ['Сони', 'сони', 'СОНИ'],
    'LG':       ['ЛГ', 'лж'],
    'Xiaomi':   ['Сяоми', 'сяоми', 'Ксиаоми'],
    'Huawei':   ['Хуавей', 'хуавей', 'Хуавэй', 'Вавей'],
    'Nokia':    ['Нокиа', 'нокия', 'Нокия'],
    'Asus':     ['Асус', 'асус', 'АСУС'],
    'Lenovo':   ['Леново', 'леново', 'Ленова'],
    'HP':       ['ХП', 'хп'],
    'Dell':     ['Дэлл', 'дэлл'],
    'Acer':     ['Асер', 'асер', 'Эйсер'],
    'Canon':    ['Кэнон', 'кэнон', 'Канон'],
    'Nikon':    ['Никон', 'никон', 'НИКОН'],
    'Bosch':    ['Бош', 'бош', 'БОШ'],
    'Philips':  ['Филипс', 'филипс', 'Филлипс'],
    'Dyson':    ['Дайсон', 'дайсон', 'Дисон'],
    'Panasonic':['Панасоник', 'панасоник'],
    'JBL':      ['ДжейБиЭл', 'джбл', 'ДжБЛ'],
    'Bose':     ['Боуз', 'боуз', 'Боз'],
    'MSI':      ['МСИ', 'мси'],
    'Logitech': ['Логитек', 'логитек', 'Лоджитек'],
    'Razer':    ['Рейзер', 'рейзер', 'Разер'],
    'Adidas':   ['Адидас', 'адидас', 'АДИДАС'],
    'Nike':     ['Найк', 'найк', 'НАЙК'],
    'Puma':     ['Пума', 'пума'],
    'Reebok':   ['Рибок', 'рибок', 'Рибок'],
    'Corsair':  ['Корсэйр', 'корсаир'],
    'Kingston': ['Кингстон', 'кингстон'],
    'WD':       ['ВД', 'вестерн диджитал'],
    'Seagate':  ['Сигейт', 'сигейт'],
}

QWERTY_NEIGHBORS: Dict[str, str] = {
    'q': 'wa',  'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg',
    'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
    'a': 'qsz', 's': 'awdxz','d': 'sefc','f': 'drgv','g': 'fthy',
    'h': 'gyjn','j': 'hukmn','k': 'jilm','l': 'kop',
    'z': 'asx', 'x': 'zsdc','c': 'xvdf','v': 'cfgb','b': 'vghn',
    'n': 'bhjm','m': 'njk',
    'й': 'цф', 'ц': 'йук', 'у': 'цке', 'к': 'уен', 'е': 'кнг',
    'н': 'егш', 'г': 'нзш', 'ш': 'гщх', 'щ': 'шхз', 'з': 'щхъ',
    'х': 'щз',  'ф': 'йыя', 'ы': 'фва', 'в': 'ыас', 'а': 'всп',
    'п': 'арол','р': 'паод','о': 'рлд', 'л': 'опж', 'д': 'лжэ',
    'ж': 'лдэ', 'э': 'жд',  'я': 'фыч', 'ч': 'яысм','с': 'чаим',
    'м': 'счит','и': 'смть','т': 'имьб','ь': 'тибю','б': 'тью',
    'ю': 'ьбж',
}

def transliterate_ru(text: str) -> str:
    result = []
    for ch in text:
        low = ch.lower()
        if low in RU_TO_LAT:
            lat = RU_TO_LAT[low]
            if ch.isupper() and lat:
                lat = lat[0].upper() + lat[1:]
            result.append(lat)
        else:
            result.append(ch)
    return ''.join(result)

def cyrillize_brand(word: str) -> str:
    variants = BRAND_TO_CYRILLIC.get(word)
    if variants:
        return random.choice(variants)
    return word

def add_char_typo(word: str, p: float = 0.12) -> str:
    if len(word) < 3:
        return word
    chars = list(word)
    i = 0
    while i < len(chars):
        if random.random() < p:
            op = random.choices(
                ['swap', 'delete', 'double', 'neighbor'],
                weights=[0.3, 0.2, 0.2, 0.3],
            )[0]
            if op == 'swap' and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 2
                continue
            elif op == 'delete':
                chars.pop(i)
                continue
            elif op == 'double':
                chars.insert(i + 1, chars[i])
                i += 2
                continue
            elif op == 'neighbor':
                neighbors = QWERTY_NEIGHBORS.get(chars[i].lower(), '')
                if neighbors:
                    replacement = random.choice(neighbors)
                    if chars[i].isupper():
                        replacement = replacement.upper()
                    chars[i] = replacement
        i += 1
    return ''.join(chars)

def random_case(word: str) -> str:
    r = random.random()
    if r < 0.4:
        return word.upper()
    elif r < 0.7:
        return word.lower()
    return word.capitalize()

def _rebuild_entities(tokens: List[str], labels: List[str]) -> Tuple[str, List[Dict]]:
    text = ' '.join(tokens)
    entities = []
    i = 0
    pos = 0
    tok_starts = []
    for tok in tokens:
        tok_starts.append(pos)
        pos += len(tok) + 1

    while i < len(labels):
        if labels[i].startswith('B-'):
            ent_type = labels[i][2:]
            start = tok_starts[i]
            j = i + 1
            while j < len(labels) and labels[j] == f'I-{ent_type}':
                j += 1
            end = tok_starts[j - 1] + len(tokens[j - 1])
            entities.append({
                'label': ent_type,
                'start': start,
                'end':   end,
                'value': text[start:end],
            })
            i = j
        else:
            i += 1
    return text, entities

def augment_record(record: Dict, strategy: str = 'mixed') -> Dict:
    tokens = list(record['tokens'])
    labels = list(record['labels'])

    if strategy == 'mixed':
        strategy = random.choices(
            ['typo', 'typo_entity', 'translit_ctx',
             'cyrillic_ent', 'translit_ent', 'case'],
            weights=[0.25, 0.10, 0.20, 0.20, 0.15, 0.10],
        )[0]

    new_tokens = []
    for tok, lbl in zip(tokens, labels):
        is_entity = lbl != 'O'

        if strategy == 'typo' and not is_entity:
            tok = add_char_typo(tok)

        elif strategy == 'typo_entity' and is_entity:
            tok = add_char_typo(tok, p=0.08)

        elif strategy == 'translit_ctx' and not is_entity:
            if sum(1 for c in tok if 'Ѐ' <= c <= 'ӿ') >= 2:
                tok = transliterate_ru(tok)

        elif strategy == 'cyrillic_ent' and is_entity:
            tok = cyrillize_brand(tok)

        elif strategy == 'translit_ent' and is_entity:
            if sum(1 for c in tok if 'Ѐ' <= c <= 'ӿ') >= 2:
                tok = transliterate_ru(tok)

        elif strategy == 'case':
            tok = random_case(tok)

        new_tokens.append(tok)

    new_text, new_entities = _rebuild_entities(new_tokens, labels)
    return {
        'text':     new_text,
        'tokens':   new_tokens,
        'labels':   labels,
        'entities': new_entities,
        'source':   record.get('source', 'synthetic') + f'_aug_{strategy}',
    }

def augment_dataset(
    data: List[Dict],
    multiplier: int = 2,
    seed: int = 42,
    strategies: List[str] = None,
) -> List[Dict]:
    random.seed(seed)
    if strategies is None:
        strategies = ['typo', 'translit_ctx', 'cyrillic_ent',
                      'translit_ent', 'case', 'mixed']

    augmented = []
    for record in data:
        for _ in range(multiplier):
            strategy = random.choice(strategies)
            augmented.append(augment_record(record, strategy))
    return augmented

if __name__ == '__main__':
    import json, os, argparse

    parser = argparse.ArgumentParser(description='Аугментация NER датасета')
    parser.add_argument('--data-dir',    default='data',      help='Папка с data/')
    parser.add_argument('--output-dir',  default='data',      help='Куда сохранять')
    parser.add_argument('--multiplier',  type=int, default=2, help='Копий на пример')
    parser.add_argument('--splits',      default='train',     help='train,val,test')
    args = parser.parse_args()

    for split in args.splits.split(','):
        src = os.path.join(args.data_dir, f'{split}.json')
        if not os.path.exists(src):
            print(f'  [!] {src} не найден — пропускаем')
            continue

        with open(src, encoding='utf-8') as f:
            original = json.load(f)

        augmented = augment_dataset(original, multiplier=args.multiplier)
        combined  = original + augmented
        random.shuffle(combined)

        out = os.path.join(args.output_dir, f'{split}_augmented.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        print(f'{split:5s}: {len(original)} orig + {len(augmented)} aug = {len(combined)} total → {out}')
