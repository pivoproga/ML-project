
import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def step_generate(num_samples: int = 3000) -> None:
    logger.info("=" * 60)
    logger.info("ШАГ 1: ГЕНЕРАЦИЯ ДАТАСЕТА")
    logger.info("=" * 60)

    from data_generator import generate_dataset, split_dataset
    from config import DATA_DIR

    ds = generate_dataset(num_samples)
    logger.info(f"Сгенерировано {len(ds)} примеров")

    train, val, test = split_dataset(ds)
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(f"{DATA_DIR}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        logger.info(f"  {name}: {len(split)} примеров → {DATA_DIR}/{name}.json")

    logger.info("\nПримеры из датасета:")
    for rec in train[:2]:
        logger.info(f"  Текст:   {rec['text']}")
        logger.info(f"  Метки:   {list(zip(rec['tokens'], rec['labels']))}")
        logger.info(f"  Сущн.:   {rec['entities']}")
        logger.info("")

def step_convert_mave(max_samples: int = 5000) -> None:
    logger.info("=" * 60)
    logger.info("ШАГ 1b: КОНВЕРТАЦИЯ MAVE + СМЕШИВАНИЕ")
    logger.info("=" * 60)

    try:
        from convert_mave import load_and_convert_mave, merge_with_synthetic, print_stats
    except ImportError as e:
        logger.error(f"Ошибка: {e}\nУстановите: pip install datasets")
        sys.exit(1)

    mave_data = load_and_convert_mave(max_samples=max_samples)
    os.makedirs("data", exist_ok=True)
    with open("data/mave_converted.json", "w", encoding="utf-8") as f:
        json.dump(mave_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Сохранено: data/mave_converted.json ({len(mave_data)} примеров)")

    print_stats(mave_data, "MAVE converted")
    merge_with_synthetic(mave_data)
    logger.info("Смешанный датасет: data/*_combined.json")

def step_train(use_combined: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ")
    logger.info("=" * 60)

    from config import DATA_DIR
    import shutil

    suffix = "_combined" if use_combined else ""
    for split in ("train", "val", "test"):
        src = f"{DATA_DIR}/{split}{suffix}.json"
        dst = f"{DATA_DIR}/{split}.json"
        if not os.path.exists(src):
            hint = "--convert-mave" if use_combined else "--generate"
            logger.error(f"Файл {src} не найден. Сначала запустите {hint}.")
            sys.exit(1)
        if use_combined:
            shutil.copy(src, dst)
            logger.info(f"  Используем {src}")

    from train import train_model
    results = train_model()

    logger.info("\n── Итоговые метрики ──")
    logger.info(f"  Лучший val F1 : {results['best_val_f1']:.4f}")
    logger.info(f"  Test F1       : {results['test_f1']:.4f}")
    logger.info(f"  Test Precision: {results['test_precision']:.4f}")
    logger.info(f"  Test Recall   : {results['test_recall']:.4f}")

def step_infer() -> None:
    logger.info("=" * 60)
    logger.info("ШАГ 3: ДЕМОНСТРАЦИЯ ИНФЕРЕНСА")
    logger.info("=" * 60)

    from config import FINAL_MODEL_DIR
    if not os.path.exists(FINAL_MODEL_DIR):
        logger.error(f"Модель не найдена: {FINAL_MODEL_DIR}. Сначала запустите --train.")
        sys.exit(1)

    from inference import run_inference
    run_inference(FINAL_MODEL_DIR)

def step_visualize() -> None:
    logger.info("=" * 60)
    logger.info("ШАГ 4: ВИЗУАЛИЗАЦИЯ")
    logger.info("=" * 60)

    from visualize import plot_training_curves, plot_entity_distribution, print_dataset_stats
    from config import OUTPUT_DIR, DATA_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_dataset_stats()

    if os.path.exists(f"{OUTPUT_DIR}/history.json"):
        plot_training_curves()
    else:
        logger.warning("history.json не найден — запустите --train сначала")

    if os.path.exists(f"{DATA_DIR}/train.json"):
        plot_entity_distribution()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NER для торговых площадок: BRAND, MODEL, PRICE, CONDITION"
    )
    parser.add_argument("--generate",      action="store_true", help="Генерация синтетических данных")
    parser.add_argument("--convert-mave",  action="store_true", help="Скачать MAVE и смешать с синтетикой")
    parser.add_argument("--train",         action="store_true", help="Обучение модели")
    parser.add_argument("--infer",         action="store_true", help="Демо-инференс")
    parser.add_argument("--visualize",     action="store_true", help="Визуализация")
    parser.add_argument("--all",           action="store_true", help="Полный пайплайн (синтетика)")
    parser.add_argument("--use-combined",  action="store_true", help="Обучать на MAVE+синтетика")
    parser.add_argument("--samples",       type=int, default=3000, help="Число синтетических примеров")
    parser.add_argument("--mave-samples",  type=int, default=5000, help="Число примеров из MAVE")
    args = parser.parse_args()

    if not any([args.generate, getattr(args, "convert_mave", False),
                args.train, args.infer, args.visualize, args.all]):
        parser.print_help()
        return

    if args.all:
        args.generate = args.train = args.infer = args.visualize = True

    if args.generate:
        step_generate(args.samples)
    if getattr(args, "convert_mave", False):
        step_convert_mave(args.mave_samples)
    if args.train:
        step_train(use_combined=getattr(args, "use_combined", False))
    if args.infer:
        step_infer()
    if args.visualize:
        step_visualize()

    logger.info("\n✓ Готово.")

if __name__ == "__main__":
    main()
