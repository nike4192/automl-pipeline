"""
Утилиты для работы с данными.

Модуль содержит функции загрузки учебных датасетов,
генерации синтетического drift и разделения данных
на reference/current для мониторинга.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_titanic_dataset() -> pd.DataFrame:
    """
    Загружает датасет Titanic из PyCaret.

    Returns:
        pd.DataFrame с данными Titanic
    """
    from pycaret.datasets import get_data
    df = get_data("titanic", verbose=False)
    logger.info(f"Загружен датасет Titanic: {df.shape[0]} строк, {df.shape[1]} колонок")
    return df


def prepare_reference_and_current(
    df: pd.DataFrame,
    reference_ratio: float = 0.7,
    inject_drift: bool = False,
    drift_magnitude: float = 2.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет датасет на reference (обучающая выборка) и current (новые данные).

    При inject_drift=True в current добавляется синтетический data drift
    для тестирования системы мониторинга.

    Args:
        df: исходный датасет
        reference_ratio: доля данных для reference
        inject_drift: добавить ли drift в current
        drift_magnitude: сила drift (множитель для числовых колонок)
        random_state: seed для воспроизводимости

    Returns:
        Tuple (reference_df, current_df)
    """
    np.random.seed(random_state)

    n_ref = int(len(df) * reference_ratio)
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    reference = shuffled.iloc[:n_ref].copy()
    current = shuffled.iloc[n_ref:].copy()

    if inject_drift:
        current = _inject_synthetic_drift(current, drift_magnitude, random_state)
        logger.info(f"Синтетический drift добавлен (magnitude={drift_magnitude})")

    logger.info(f"Reference: {reference.shape[0]} строк, Current: {current.shape[0]} строк")
    return reference, current


def _inject_synthetic_drift(
    df: pd.DataFrame,
    magnitude: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Вносит синтетический drift в числовые колонки.

    Сдвигает средние значения числовых признаков на magnitude * std,
    имитируя реальный data drift в production.
    """
    np.random.seed(random_state)
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_std = df[col].std()
        if col_std > 0:
            # Сдвиг среднего + добавление шума
            shift = magnitude * col_std * np.random.choice([-1, 1])
            noise = np.random.normal(0, col_std * 0.3, size=len(df))
            df[col] = df[col] + shift + noise

    return df


def save_datasets(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[str, str]:
    """
    Сохраняет reference и current датасеты в data/.

    Returns:
        Tuple (путь к reference.csv, путь к current.csv)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ref_path = DATA_DIR / "reference.csv"
    cur_path = DATA_DIR / "current.csv"

    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)

    logger.info(f"Сохранены: {ref_path}, {cur_path}")
    return str(ref_path), str(cur_path)


def setup_initial_data(inject_drift: bool = True) -> tuple[str, str]:
    """
    Полный цикл подготовки данных: загрузка → разделение → сохранение.

    Точка входа для инициализации проекта.

    Args:
        inject_drift: добавить drift в current (True для демонстрации)

    Returns:
        Tuple (путь к reference.csv, путь к current.csv)
    """
    df = load_titanic_dataset()
    reference, current = prepare_reference_and_current(
        df, inject_drift=inject_drift
    )
    return save_datasets(reference, current)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ref_path, cur_path = setup_initial_data(inject_drift=True)
    print(f"Reference: {ref_path}")
    print(f"Current:   {cur_path}")
