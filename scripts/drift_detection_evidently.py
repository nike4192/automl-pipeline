"""
Детекция Data Drift с использованием Evidently AI.

Скрипт выполняет:
1. Загрузку reference и current данных (Titanic dataset)
2. Генерацию синтетического drift для демонстрации
3. Построение Data Drift Report (Evidently)
4. Построение Data Quality Report
5. Вычисление PSI и KS-тест вручную для сравнения
6. Сохранение HTML-отчётов и метрик

Результаты сохраняются в reports/
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_titanic_data() -> pd.DataFrame:
    """Загружает Titanic dataset из seaborn или CSV."""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        # Оставляем только числовые и ключевые признаки
        df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
        df.columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    except ImportError:
        # Fallback: генерируем синтетические данные
        logger.warning("seaborn не найден, генерируем синтетические данные")
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            "Survived": np.random.binomial(1, 0.38, n),
            "Pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
            "Age": np.random.normal(29.7, 14.5, n).clip(0.5, 80),
            "SibSp": np.random.poisson(0.5, n).clip(0, 8),
            "Parch": np.random.poisson(0.4, n).clip(0, 6),
            "Fare": np.random.exponential(32, n).clip(0, 512),
            "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
        })

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df


def create_reference_and_current(df: pd.DataFrame, drift_strength: float = 0.3):
    """
    Разделяет данные на reference и current, добавляя синтетический drift.

    Args:
        df: исходный датасет
        drift_strength: сила drift (0 = нет, 1 = максимальный)

    Returns:
        reference, current — два DataFrame
    """
    np.random.seed(42)

    # 70/30 split
    split_idx = int(len(df) * 0.7)
    reference = df.iloc[:split_idx].copy().reset_index(drop=True)
    current = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Добавляем drift к current
    logger.info(f"Добавляем синтетический drift (strength={drift_strength})")

    # Age: сдвиг среднего на +10 лет
    current["Age"] = current["Age"] + drift_strength * 10

    # Fare: увеличение на 50%
    current["Fare"] = current["Fare"] * (1 + drift_strength * 0.5)

    # Pclass: больше 1-го класса
    mask = np.random.random(len(current)) < drift_strength * 0.3
    current.loc[mask, "Pclass"] = 1

    # SibSp: небольшой сдвиг
    current["SibSp"] = (current["SibSp"] + np.random.binomial(1, drift_strength * 0.2, len(current))).clip(0, 8)

    return reference, current


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Вычисляет PSI (Population Stability Index)."""
    eps = 1e-4
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )
    ref_pct = np.histogram(reference.dropna(), bins=breakpoints)[0] / len(reference.dropna()) + eps
    cur_pct = np.histogram(current.dropna(), bins=breakpoints)[0] / len(current.dropna()) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_manual_drift_tests(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Ручной расчёт PSI и KS-тест для числовых признаков."""
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    if "Survived" in numeric_cols:
        numeric_cols.remove("Survived")

    results = {}
    for col in numeric_cols:
        psi = calculate_psi(reference[col], current[col])
        ks_stat, ks_pvalue = stats.ks_2samp(reference[col].dropna(), current[col].dropna())
        results[col] = {
            "psi": round(psi, 4),
            "psi_drift": psi > 0.2,
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 6),
            "ks_drift": ks_pvalue < 0.05,
            "ref_mean": round(reference[col].mean(), 4),
            "cur_mean": round(current[col].mean(), 4),
            "mean_shift": round(abs(current[col].mean() - reference[col].mean()), 4),
        }

    return results


def build_evidently_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Строит Evidently Data Drift Report."""
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    report.run(reference_data=reference, current_data=current)
    return report


def build_evidently_quality_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Строит Evidently Data Quality Report."""
    report = Report(metrics=[
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    return report


def build_column_drift_reports(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Детальный drift-отчёт по каждому числовому признаку."""
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    if "Survived" in numeric_cols:
        numeric_cols.remove("Survived")

    metrics = [ColumnDriftMetric(column_name=col) for col in numeric_cols]
    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current)
    return report


def main():
    """Основной скрипт детекции data drift."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ДЕТЕКЦИЯ DATA DRIFT — Titanic Dataset")
    logger.info("=" * 60)

    # 1. Загрузка данных
    logger.info("\n[1/6] Загрузка данных...")
    df = load_titanic_data()
    logger.info(f"  Загружено {len(df)} записей, {len(df.columns)} признаков")

    # 2. Создание reference и current с drift
    logger.info("\n[2/6] Генерация reference/current с синтетическим drift...")
    reference, current = create_reference_and_current(df, drift_strength=0.3)
    logger.info(f"  Reference: {len(reference)} записей")
    logger.info(f"  Current: {len(current)} записей")

    # Сохраняем данные
    reference.to_csv(DATA_DIR / "reference.csv", index=False)
    current.to_csv(DATA_DIR / "current.csv", index=False)
    logger.info("  Данные сохранены в data/")

    # 3. Ручной расчёт PSI и KS-тест
    logger.info("\n[3/6] Ручной расчёт PSI и KS-тест...")
    manual_results = run_manual_drift_tests(reference, current)

    print("\n" + "=" * 70)
    print(f"{'Признак':<12} {'PSI':>8} {'PSI Drift':>10} {'KS Stat':>10} {'KS p-val':>10} {'KS Drift':>10}")
    print("-" * 70)
    for col, r in manual_results.items():
        print(f"{col:<12} {r['psi']:>8.4f} {'YES' if r['psi_drift'] else 'no':>10} "
              f"{r['ks_statistic']:>10.4f} {r['ks_pvalue']:>10.6f} {'YES' if r['ks_drift'] else 'no':>10}")
    print("=" * 70)

    drift_detected = any(r["psi_drift"] for r in manual_results.values())
    max_psi = max(r["psi"] for r in manual_results.values())
    n_drifted = sum(1 for r in manual_results.values() if r["psi_drift"])
    print(f"\nОбщий результат: Drift {'ОБНАРУЖЕН' if drift_detected else 'НЕ обнаружен'}")
    print(f"Макс. PSI: {max_psi:.4f} (порог: 0.2)")
    print(f"Признаков с drift: {n_drifted}/{len(manual_results)}")

    # Сохраняем JSON с результатами
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "max_psi": max_psi,
        "n_features_drifted": n_drifted,
        "n_features_total": len(manual_results),
        "features": manual_results,
    }
    with open(REPORTS_DIR / "drift_results.json", "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info("  Результаты сохранены в reports/drift_results.json")

    # 4. Evidently Data Drift Report
    logger.info("\n[4/6] Построение Evidently Data Drift Report...")
    drift_report = build_evidently_drift_report(reference, current)
    drift_report.save_html(str(REPORTS_DIR / "data_drift_report.html"))
    logger.info("  Сохранён: reports/data_drift_report.html")

    # 5. Evidently Data Quality Report
    logger.info("\n[5/6] Построение Evidently Data Quality Report...")
    quality_report = build_evidently_quality_report(reference, current)
    quality_report.save_html(str(REPORTS_DIR / "data_quality_report.html"))
    logger.info("  Сохранён: reports/data_quality_report.html")

    # 6. Детальный drift по признакам
    logger.info("\n[6/6] Построение детального Column Drift Report...")
    column_report = build_column_drift_reports(reference, current)
    column_report.save_html(str(REPORTS_DIR / "column_drift_report.html"))
    logger.info("  Сохранён: reports/column_drift_report.html")

    # Итог
    logger.info("\n" + "=" * 60)
    logger.info("ГОТОВО! Все отчёты сохранены в reports/")
    logger.info("=" * 60)
    print("\nОтчёты:")
    print(f"  1. reports/drift_results.json     — метрики PSI/KS (JSON)")
    print(f"  2. reports/data_drift_report.html  — Evidently Drift Report")
    print(f"  3. reports/data_quality_report.html — Evidently Quality Report")
    print(f"  4. reports/column_drift_report.html — Drift по признакам")

    return results_json


if __name__ == "__main__":
    main()
