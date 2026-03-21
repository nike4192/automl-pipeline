"""
PyCaret AutoML модель для автоматического обучения и выбора лучшей модели.

Этап 2 ML-процесса: автоматический отбор и обучение модели через PyCaret
с автологированием экспериментов в MLflow.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv
from pycaret.classification import (
    setup,
    compare_models,
    create_model,
    save_model,
    pull
)

# Загрузка переменных окружения
load_dotenv()

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Загрузить и подготовить данные для обучения.

    Args:
        data_path: Путь к CSV файлу с обучающими данными

    Returns:
        DataFrame с подготовленными данными

    Raises:
        FileNotFoundError: Если файл данных не найден
        ValueError: Если данные пусты или некорректны
    """
    try:
        logger.info(f"Загрузка данных из {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Загруженный датасет пуст")

        logger.info(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
        return df

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        raise


def train_and_save() -> Dict[str, Any]:
    """
    Основная функция: подготовка данных, PyCaret setup, выбор лучшей модели и сохранение.

    Этапы:
    1. Загрузить данные из data/reference.csv
    2. Инициализировать PyCaret setup с log_experiment=True (автологирование в MLflow)
    3. Запустить compare_models() для перебора алгоритмов
    4. Сохранить лучшую модель
    5. Вернуть информацию о лучшей модели

    Returns:
        Dict с информацией о лучшей модели:
        {
            'model_name': str,           # Имя алгоритма (e.g., 'Linear Regression')
            'model_type': str,           # Тип модели (e.g., 'lr')
            'metrics': Dict,             # Метрики на тестовом наборе
            'saved_model_path': str      # Путь сохранённой модели
        }

    Raises:
        Exception: При ошибках на любом этапе
    """
    try:
        # ============================================================================
        # 1. Загрузка и подготовка данных
        # ============================================================================
        data_path = os.getenv('DATA_PATH', 'data/reference.csv')
        df = load_and_prepare_data(data_path)

        # Получить целевую переменную из .env (по умолчанию "Survived" для Titanic)
        target_column = os.getenv('TARGET_COLUMN', 'Survived')

        if target_column not in df.columns:
            raise ValueError(
                f"Целевая переменная '{target_column}' не найдена в данных. "
                f"Доступные столбцы: {df.columns.tolist()}"
            )

        logger.info(f"Целевая переменная: {target_column}")
        logger.info(f"Распределение целевой переменной:\n{df[target_column].value_counts()}")

        # ============================================================================
        # 2. PyCaret Setup — инициализация и подготовка к AutoML
        # ============================================================================
        # log_experiment=True — автоматическое логирование экспериментов в MLflow
        # Это критически важно для интеграции с MLflow Registry
        logger.info("Инициализация PyCaret с параметром log_experiment=True...")

        setup(
            data=df,
            target=target_column,
            log_experiment=True,  # КРИТИЧНО: автологирование в MLflow
            experiment_name='automl_titanic',
            session_id=42,
            verbose=False
        )

        logger.info("PyCaret успешно инициализирован")

        # ============================================================================
        # 3. Поиск и выбор лучшей модели
        # ============================================================================
        # compare_models() автоматически тестирует множество алгоритмов:
        # LR, KNN, NB, DT, RF, SVM, XGB, LightGBM и др.
        # Возвращает лучшую модель по метрике (по умолчанию — Accuracy)
        logger.info("Запуск compare_models() для поиска лучшей модели...")
        logger.info("Это может занять несколько минут...")

        best_model = compare_models(sort='AUC')  # Сортировка по AUC для better ranking

        logger.info(f"Найдена лучшая модель: {type(best_model).__name__}")

        # ============================================================================
        # 3.1 Обучение второй модели для A/B теста (Staging)
        # ============================================================================
        # compare_models возвращает лучшую модель, но для A/B теста нужна
        # вторая обученная модель. Обучим Logistic Regression как альтернативу.
        logger.info("Обучение второй модели (Logistic Regression) для A/B теста...")
        second_model = create_model('lr')
        logger.info(f"Вторая модель обучена: {type(second_model).__name__}")

        # ============================================================================
        # 4. Сохранение моделей
        # ============================================================================
        model_name = 'best_automl_model'
        second_model_name = 'staging_automl_model'

        logger.info(f"Сохранение лучшей модели как '{model_name}'...")
        saved_model = save_model(best_model, model_name=model_name)
        model_path = f"{model_name}.pkl"
        logger.info(f"Модель сохранена: {model_path}")

        logger.info(f"Сохранение второй модели как '{second_model_name}'...")
        save_model(second_model, model_name=second_model_name)
        logger.info(f"Вторая модель сохранена: {second_model_name}.pkl")

        # ============================================================================
        # 5. Извлечение метрик из последнего эксперимента
        # ============================================================================
        # pull() возвращает историю экспериментов PyCaret
        # Используем это для документирования результатов
        logger.info("Извлечение метрик обучения...")

        experiment_results = pull()

        # Собрать ключевые метрики из результатов
        metrics = {
            'model_type': str(type(best_model).__name__),
            'accuracy': experiment_results.iloc[0]['Accuracy'] if 'Accuracy' in experiment_results.columns else 0,
            'auc': experiment_results.iloc[0]['AUC'] if 'AUC' in experiment_results.columns else 0,
            'precision': experiment_results.iloc[0]['Precision'] if 'Precision' in experiment_results.columns else 0,
            'recall': experiment_results.iloc[0]['Recall'] if 'Recall' in experiment_results.columns else 0,
            'f1': experiment_results.iloc[0]['F1'] if 'F1' in experiment_results.columns else 0,
        }

        logger.info(f"Основные метрики модели:\n{metrics}")

        # ============================================================================
        # 6. Возврат результатов
        # ============================================================================
        result = {
            'model_name': model_name,
            'model_type': str(type(best_model).__name__),
            'metrics': metrics,
            'saved_model_path': model_path
        }

        logger.info("Завершение AutoML обучения успешно")
        return result

    except Exception as e:
        logger.error(f"Критическая ошибка при обучении модели: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    """
    Точка входа для запуска AutoML обучения.

    Используется в Airflow DAG для запуска переобучения модели.
    """
    try:
        logger.info("=" * 80)
        logger.info("ЗАПУСК AUTOML ОБУЧЕНИЯ")
        logger.info("=" * 80)

        result = train_and_save()

        logger.info("=" * 80)
        logger.info("РЕЗУЛЬТАТЫ AUTOML")
        logger.info("=" * 80)
        logger.info(f"Сохранённая модель: {result['saved_model_path']}")
        logger.info(f"Тип модели: {result['model_type']}")
        logger.info(f"Метрики:\n{result['metrics']}")

    except Exception as e:
        logger.error(f"Неудача при выполнении AutoML: {str(e)}")
        exit(1)
