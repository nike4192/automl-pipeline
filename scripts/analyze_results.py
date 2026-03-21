"""
Анализ результатов A/B теста — вычисление метрик и статистическая проверка.

Этап 5 ML-процесса: анализ логов A/B теста, вычисление ML и бизнес-метрик,
проведение статистических тестов для определения, готова ли новая модель к Production.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ab_test_logs(log_path: str = 'logs/ab_test_log.csv') -> pd.DataFrame:
    """
    Загрузить логи A/B теста из CSV.

    Args:
        log_path: Путь к файлу логов

    Returns:
        DataFrame с логами A/B теста

    Raises:
        FileNotFoundError: Если файл логов не найден
    """
    try:
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Файл логов не найден: {log_path}")

        df = pd.read_csv(log_path)
        logger.info(f"Загружено {len(df)} записей A/B теста из {log_path}")

        return df

    except Exception as e:
        logger.error(f"Ошибка при загрузке логов: {str(e)}")
        raise


def compute_ml_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Вычислить ML метрики для каждой модели (A и B).

    Вычисляет метрики отдельно для каждой модели, если в логах есть целевое значение.

    Метрики:
    - Accuracy: доля правильных предсказаний
    - Precision: доля положительных предсказаний, которые верны
    - Recall: доля реальных положительных случаев, которые найдены
    - F1-score: гармоническое среднее Precision и Recall
    - ROC-AUC: площадь под кривой ROC

    Args:
        df: DataFrame с логами A/B теста

    Returns:
        Dict с метриками для каждой модели:
        {
            'model_A': {'accuracy': float, 'precision': float, ...},
            'model_B': {'accuracy': float, 'precision': float, ...}
        }
    """
    try:
        logger.info("Вычисление ML метрик...")

        # ================================================================
        # 1. Проверить наличие необходимых колонок
        # ================================================================
        required_cols = ['model_stage', 'prediction']
        if 'true_label' in df.columns:
            required_cols.append('true_label')

        metrics = {}

        # ================================================================
        # 2. Вычислить метрики для каждой модели
        # ================================================================
        for stage in df['model_stage'].unique():
            try:
                stage_data = df[df['model_stage'] == stage]

                if len(stage_data) == 0:
                    continue

                logger.info(f"\nМетрики для {stage} (n={len(stage_data)}):")

                # ====================================================
                # Если есть истинные метки
                # ====================================================
                if 'true_label' in stage_data.columns:
                    y_true = stage_data['true_label']
                    y_pred = stage_data['prediction']

                    # Accuracy
                    accuracy = (y_true == y_pred).mean()

                    # Precision, Recall, F1 для бинарной классификации
                    if len(stage_data['prediction'].unique()) == 2:
                        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)

                        try:
                            roc_auc = roc_auc_score(y_true, y_pred)
                        except:
                            roc_auc = None
                    else:
                        precision = None
                        recall = None
                        f1 = None
                        roc_auc = None

                    metrics[stage] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision) if precision is not None else None,
                        'recall': float(recall) if recall is not None else None,
                        'f1': float(f1) if f1 is not None else None,
                        'roc_auc': float(roc_auc) if roc_auc is not None else None,
                        'n_samples': len(stage_data)
                    }

                    logger.info(f"  Accuracy: {accuracy:.4f}")
                    if precision is not None:
                        logger.info(f"  Precision: {precision:.4f}")
                    if recall is not None:
                        logger.info(f"  Recall: {recall:.4f}")
                    if f1 is not None:
                        logger.info(f"  F1-score: {f1:.4f}")

                else:
                    # Если нет истинных меток, вычислить только статистику предсказаний
                    metrics[stage] = {
                        'prediction_dist': stage_data['prediction'].value_counts().to_dict(),
                        'n_samples': len(stage_data)
                    }

            except Exception as e:
                logger.warning(f"Ошибка при вычислении метрик для {stage}: {str(e)}")
                continue

        return metrics

    except Exception as e:
        logger.error(f"Ошибка при вычислении ML метрик: {str(e)}", exc_info=True)
        raise


def compute_business_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Вычислить бизнес-метрики на основе логов A/B теста.

    Метрики:
    - Среднее время ответа (response_time_ms)
    - Распределение предсказаний по классам
    - Пропускная способность (requests/sec)

    Args:
        df: DataFrame с логами A/B теста

    Returns:
        Dict с бизнес-метриками
    """
    try:
        logger.info("Вычисление бизнес-метрик...")

        metrics = {}

        # ================================================================
        # 1. Время ответа
        # ================================================================
        if 'response_time_ms' in df.columns:
            metrics['avg_response_time_ms'] = float(df['response_time_ms'].mean())
            metrics['p50_response_time_ms'] = float(df['response_time_ms'].quantile(0.5))
            metrics['p95_response_time_ms'] = float(df['response_time_ms'].quantile(0.95))
            metrics['p99_response_time_ms'] = float(df['response_time_ms'].quantile(0.99))

            logger.info(f"  Avg response time: {metrics['avg_response_time_ms']:.2f}ms")
            logger.info(f"  P95 response time: {metrics['p95_response_time_ms']:.2f}ms")

        # ================================================================
        # 2. Распределение предсказаний
        # ================================================================
        if 'prediction' in df.columns:
            pred_dist = df['prediction'].value_counts()
            metrics['prediction_distribution'] = pred_dist.to_dict()
            logger.info(f"  Распределение предсказаний: {pred_dist.to_dict()}")

        # ================================================================
        # 3. Трафик по моделям
        # ================================================================
        if 'model_stage' in df.columns:
            traffic_dist = df['model_stage'].value_counts()
            metrics['traffic_distribution'] = traffic_dist.to_dict()
            logger.info(f"  Распределение трафика: {traffic_dist.to_dict()}")

        return metrics

    except Exception as e:
        logger.error(f"Ошибка при вычислении бизнес-метрик: {str(e)}", exc_info=True)
        raise


def statistical_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Провести статистические тесты для сравнения моделей A и B.

    Тесты:
    1. T-тест на различие средних времён ответа
    2. Chi-square тест на различие распределений предсказаний

    Args:
        df: DataFrame с логами A/B теста

    Returns:
        Dict с результатами статистических тестов и выводами
    """
    try:
        logger.info("Проведение статистических тестов...")

        results = {}

        # ================================================================
        # 1. T-тест на различие response time
        # ================================================================
        if 'response_time_ms' in df.columns and 'model_stage' in df.columns:
            logger.info("\n[T-тест] Сравнение среднего времени ответа:")

            stages = df['model_stage'].unique()
            if len(stages) >= 2:
                stage_a = df[df['model_stage'] == stages[0]]['response_time_ms']
                stage_b = df[df['model_stage'] == stages[1]]['response_time_ms']

                t_stat, p_value = stats.ttest_ind(stage_a, stage_b)

                results['response_time_ttest'] = {
                    'model_a_mean': float(stage_a.mean()),
                    'model_b_mean': float(stage_b.mean()),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }

                logger.info(f"  Model A mean: {stage_a.mean():.2f}ms")
                logger.info(f"  Model B mean: {stage_b.mean():.2f}ms")
                logger.info(f"  t-statistic: {t_stat:.4f}")
                logger.info(f"  p-value: {p_value:.6f}")
                logger.info(
                    f"  Результат: {'ЗНАЧИМОЕ различие' if p_value < 0.05 else 'различие незначимо'} "
                    f"(α=0.05)"
                )

        # ================================================================
        # 2. Chi-square тест на различие распределений
        # ================================================================
        if 'prediction' in df.columns and 'model_stage' in df.columns:
            logger.info("\n[Chi-square тест] Сравнение распределений предсказаний:")

            # Построить contingency table
            contingency = pd.crosstab(
                df['model_stage'],
                df['prediction']
            )

            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            results['prediction_distribution_chi2'] = {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }

            logger.info(f"  Chi-square statistic: {chi2:.4f}")
            logger.info(f"  p-value: {p_value:.6f}")
            logger.info(f"  Degrees of freedom: {dof}")
            logger.info(
                f"  Результат: {'ЗНАЧИМОЕ различие' if p_value < 0.05 else 'различие незначимо'} "
                f"в распределениях (α=0.05)"
            )

        return results

    except Exception as e:
        logger.error(f"Ошибка при проведении статистических тестов: {str(e)}")
        return {}


def generate_visualizations(df: pd.DataFrame, output_dir: str = 'logs'):
    """
    Создать визуализации для анализа результатов A/B теста.

    Графики:
    1. Распределение времени ответа по моделям
    2. Распределение предсказаний
    3. Трафик по моделям во времени

    Args:
        df: DataFrame с логами A/B теста
        output_dir: Директория для сохранения графиков
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Создание визуализаций в {output_dir}...")

        # ================================================================
        # 1. Распределение response time
        # ================================================================
        if 'response_time_ms' in df.columns and 'model_stage' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            for stage in df['model_stage'].unique():
                stage_data = df[df['model_stage'] == stage]['response_time_ms']
                ax.hist(stage_data, alpha=0.6, label=stage, bins=30)

            ax.set_xlabel('Response Time (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Response Times by Model')
            ax.legend()
            ax.grid(True, alpha=0.3)

            path = os.path.join(output_dir, 'response_time_distribution.png')
            fig.savefig(path, dpi=100, bbox_inches='tight')
            logger.info(f"  Сохранён график: {path}")
            plt.close()

        # ================================================================
        # 2. Распределение предсказаний
        # ================================================================
        if 'prediction' in df.columns and 'model_stage' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            pred_counts = df.groupby(['model_stage', 'prediction']).size().unstack(fill_value=0)
            pred_counts.plot(kind='bar', ax=ax)

            ax.set_xlabel('Model Stage')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Predictions by Model')
            ax.legend(title='Prediction')
            ax.grid(True, alpha=0.3, axis='y')

            path = os.path.join(output_dir, 'prediction_distribution.png')
            fig.savefig(path, dpi=100, bbox_inches='tight')
            logger.info(f"  Сохранён график: {path}")
            plt.close()

        # ================================================================
        # 3. Трафик по времени
        # ================================================================
        if 'timestamp' in df.columns and 'model_stage' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            traffic_time = df.groupby([pd.Grouper(key='timestamp', freq='1H'), 'model_stage']).size().unstack(fill_value=0)

            traffic_time.plot(ax=ax, marker='o')

            ax.set_xlabel('Time')
            ax.set_ylabel('Request Count')
            ax.set_title('Traffic Distribution Over Time')
            ax.legend(title='Model Stage')
            ax.grid(True, alpha=0.3)

            path = os.path.join(output_dir, 'traffic_over_time.png')
            fig.savefig(path, dpi=100, bbox_inches='tight')
            logger.info(f"  Сохранён график: {path}")
            plt.close()

    except Exception as e:
        logger.warning(f"Ошибка при создании визуализаций: {str(e)}")


def generate_report(
    ml_metrics: Dict[str, Dict],
    business_metrics: Dict[str, Any],
    statistical_results: Dict[str, Any],
    output_path: str = 'logs/ab_test_report.txt'
) -> str:
    """
    Сгенерировать текстовый отчёт с результатами A/B теста.

    Args:
        ml_metrics: ML метрики по моделям
        business_metrics: Бизнес-метрики
        statistical_results: Результаты статистических тестов
        output_path: Путь для сохранения отчёта

    Returns:
        str: Содержимое отчёта
    """
    try:
        logger.info(f"Генерация отчёта в {output_path}...")

        report = []
        report.append("=" * 80)
        report.append("A/B ТЕСТ ОТЧЁТ")
        report.append("=" * 80)
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ================================================================
        # 1. ML метрики
        # ================================================================
        report.append("\n[ML МЕТРИКИ]\n")
        report.append("-" * 80)

        for model, metrics in ml_metrics.items():
            report.append(f"\n{model}:")
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    if isinstance(metric_value, dict):
                        report.append(f"  {metric_name}: {metric_value}")
                    else:
                        report.append(f"  {metric_name}: {metric_value:.4f}")

        # ================================================================
        # 2. Бизнес-метрики
        # ================================================================
        report.append("\n\n[БИЗНЕС-МЕТРИКИ]\n")
        report.append("-" * 80)

        for metric_name, metric_value in business_metrics.items():
            if isinstance(metric_value, dict):
                report.append(f"\n{metric_name}:")
                for k, v in metric_value.items():
                    report.append(f"  {k}: {v}")
            else:
                report.append(f"\n{metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"\n{metric_name}: {metric_value}")

        # ================================================================
        # 3. Статистические результаты
        # ================================================================
        report.append("\n\n[СТАТИСТИЧЕСКИЕ ТЕСТЫ]\n")
        report.append("-" * 80)

        for test_name, test_results in statistical_results.items():
            report.append(f"\n{test_name}:")
            for result_name, result_value in test_results.items():
                report.append(f"  {result_name}: {result_value}")

        # ================================================================
        # 4. Вывод
        # ================================================================
        report.append("\n\n[ЗАКЛЮЧЕНИЕ]\n")
        report.append("-" * 80)

        can_promote = True

        # Проверить результаты статистических тестов
        for test_results in statistical_results.values():
            if 'significant' in test_results and test_results['significant']:
                logger.info("Найдены значимые различия между моделями")
                can_promote = False

        if can_promote:
            report.append(
                "\n✅ РЕКОМЕНДАЦИЯ: Новую модель можно переводить в Production\n"
                "   (Статистически значимых различий между моделями не обнаружено)"
            )
        else:
            report.append(
                "\n⚠️  РЕКОМЕНДАЦИЯ: Требуется дополнительное тестирование\n"
                "   (Обнаружены значимые различия между моделями)"
            )

        report.append("\n" + "=" * 80)

        # Сохранить отчёт
        report_text = '\n'.join(report)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Отчёт сохранён: {output_path}")

        return report_text

    except Exception as e:
        logger.error(f"Ошибка при генерации отчёта: {str(e)}")
        raise


def run_analysis(
    log_path: str = 'logs/ab_test_log.csv',
    report_path: str = 'logs/ab_test_report.txt'
) -> Dict[str, Any]:
    """
    Основная функция для анализа результатов A/B теста.

    Этапы:
    1. Загрузить логи A/B теста
    2. Вычислить ML метрики
    3. Вычислить бизнес-метрики
    4. Провести статистические тесты
    5. Создать визуализации
    6. Сгенерировать отчёт

    Args:
        log_path: Путь к логам A/B теста
        report_path: Путь для сохранения отчёта

    Returns:
        Dict со всеми результатами анализа
    """
    try:
        logger.info("=" * 80)
        logger.info("НАЧАЛО АНАЛИЗА A/B ТЕСТА")
        logger.info("=" * 80)

        # 1. Загрузить логи
        df = load_ab_test_logs(log_path)

        # 2. Вычислить метрики
        ml_metrics = compute_ml_metrics(df)
        business_metrics = compute_business_metrics(df)

        # 3. Статистические тесты
        statistical_results = statistical_tests(df)

        # 4. Визуализации
        generate_visualizations(df)

        # 5. Отчёт
        report_text = generate_report(
            ml_metrics,
            business_metrics,
            statistical_results,
            report_path
        )

        logger.info("=" * 80)
        logger.info("АНАЛИЗ ЗАВЕРШЁН")
        logger.info("=" * 80)

        return {
            'ml_metrics': ml_metrics,
            'business_metrics': business_metrics,
            'statistical_results': statistical_results,
            'report': report_text
        }

    except Exception as e:
        logger.error(f"Критическая ошибка при анализе: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    """
    Точка входа для запуска анализа A/B теста.
    """
    try:
        run_analysis()
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        exit(1)
