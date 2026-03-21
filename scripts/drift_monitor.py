"""
Модуль мониторинга Data Drift.

Реализует три метода обнаружения drift:
1. PSI (Population Stability Index)
2. KS-тест (Kolmogorov-Smirnov)
3. Сравнение средних (change in mean/median)

При превышении threshold генерирует сигнал для запуска retraining.
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Вычисляет Population Stability Index (PSI).

    PSI измеряет сдвиг распределения между reference и current данными.
    Интерпретация:
        PSI < 0.1  — нет значительного drift
        PSI 0.1-0.2 — умеренный drift, требует внимания
        PSI > 0.2  — значительный drift, необходим retraining

    Args:
        reference: эталонное распределение
        current: текущее распределение
        bins: количество бинов для дискретизации
        eps: малая добавка для избежания деления на 0

    Returns:
        Значение PSI (float)
    """
    # Определяем границы бинов по reference
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )

    # Считаем доли в каждом бине
    ref_counts = np.histogram(reference.dropna(), bins=breakpoints)[0]
    cur_counts = np.histogram(current.dropna(), bins=breakpoints)[0]

    ref_pct = ref_counts / len(reference.dropna()) + eps
    cur_pct = cur_counts / len(current.dropna()) + eps

    # PSI = Σ (cur_pct - ref_pct) * ln(cur_pct / ref_pct)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def calculate_ks_test(
    reference: pd.Series,
    current: pd.Series,
) -> dict:
    """
    Выполняет тест Колмогорова-Смирнова для обнаружения drift.

    Returns:
        dict с ключами: statistic, p_value, is_drift
    """
    stat, p_value = stats.ks_2samp(
        reference.dropna().values,
        current.dropna().values,
    )
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_drift": p_value < 0.05,
    }


def calculate_mean_shift(
    reference: pd.Series,
    current: pd.Series,
    threshold_std: float = 1.0,
) -> dict:
    """
    Проверяет сдвиг среднего значения.

    Drift фиксируется, если |mean_current - mean_reference| > threshold_std * std_reference.

    Returns:
        dict с ключами: ref_mean, cur_mean, shift, threshold, is_drift
    """
    ref_mean = reference.mean()
    cur_mean = current.mean()
    ref_std = reference.std()

    shift = abs(cur_mean - ref_mean)
    threshold = threshold_std * ref_std

    return {
        "ref_mean": float(ref_mean),
        "cur_mean": float(cur_mean),
        "shift": float(shift),
        "threshold": float(threshold),
        "is_drift": shift > threshold,
    }


def check_drift(
    reference_path: str,
    current_path: str,
    psi_threshold: float = None,
) -> dict:
    """
    Комплексная проверка Data Drift по всем числовым признакам.

    Применяет PSI, KS-тест и сравнение средних к каждому числовому столбцу.
    Общее решение: drift обнаружен, если PSI хотя бы одного признака > threshold.

    Args:
        reference_path: путь к reference.csv
        current_path: путь к current.csv
        psi_threshold: порог PSI (по умолчанию из .env)

    Returns:
        dict с результатами проверки и флагом drift_detected
    """
    if psi_threshold is None:
        psi_threshold = float(os.getenv("DRIFT_THRESHOLD_PSI", 0.2))

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Числовые колонки (без target)
    target = os.getenv("TARGET_COLUMN", "target")
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    results = {"features": {}, "timestamp": datetime.now().isoformat()}
    max_psi = 0.0

    for col in numeric_cols:
        if col not in current.columns:
            continue

        psi = calculate_psi(reference[col], current[col])
        ks = calculate_ks_test(reference[col], current[col])
        mean_shift = calculate_mean_shift(reference[col], current[col])

        results["features"][col] = {
            "psi": psi,
            "ks_test": ks,
            "mean_shift": mean_shift,
        }
        max_psi = max(max_psi, psi)

        logger.info(
            f"  {col}: PSI={psi:.4f}, KS_stat={ks['statistic']:.4f}, "
            f"KS_drift={ks['is_drift']}, mean_shift={mean_shift['is_drift']}"
        )

    results["max_psi"] = max_psi
    results["psi_threshold"] = psi_threshold
    results["drift_detected"] = max_psi > psi_threshold

    # Количество признаков с drift по каждому методу
    n_psi_drift = sum(
        1 for f in results["features"].values() if f["psi"] > psi_threshold
    )
    n_ks_drift = sum(
        1 for f in results["features"].values() if f["ks_test"]["is_drift"]
    )
    results["n_features_psi_drift"] = n_psi_drift
    results["n_features_ks_drift"] = n_ks_drift

    # Сохраняем лог
    _save_drift_log(results)

    level = logging.WARNING if results["drift_detected"] else logging.INFO
    logger.log(
        level,
        f"Drift check: max_PSI={max_psi:.4f}, threshold={psi_threshold}, "
        f"drift_detected={results['drift_detected']}",
    )

    return results


def _save_drift_log(results: dict) -> None:
    """Сохраняет результаты drift-проверки в logs/drift_history.jsonl."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "drift_history.jsonl"

    # Сериализуемая версия
    log_entry = {
        "timestamp": results["timestamp"],
        "max_psi": results["max_psi"],
        "threshold": results["psi_threshold"],
        "drift_detected": results["drift_detected"],
        "n_features_psi_drift": results["n_features_psi_drift"],
        "n_features_ks_drift": results["n_features_ks_drift"],
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = check_drift(
        reference_path=str(PROJECT_ROOT / "data" / "reference.csv"),
        current_path=str(PROJECT_ROOT / "data" / "current.csv"),
    )
    print(f"\nDrift detected: {result['drift_detected']}")
    print(f"Max PSI: {result['max_psi']:.4f}")
    print(f"Threshold: {result['psi_threshold']}")
