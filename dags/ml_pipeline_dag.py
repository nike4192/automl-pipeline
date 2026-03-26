"""
Apache Airflow DAG для автоматического ML pipeline переобучения.

Этап 1-5 ML-процесса: оркестрация всех этапов от мониторинга drift до A/B теста.

DAG структура:
1. check_data_drift — проверка наличия data drift
2. train_automl (условный) — обучение модели если обнаружен drift
3. register_model (условный) — регистрация в MLflow Registry
4. notify_complete — логирование завершения
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator

# ============================================================================
# ИМПОРТИРУЕМ МОДУЛИ НАШЕГО ПРОЕКТА
# ============================================================================
# Примечание: в production окружении используйте полные пути или добавьте
# в PYTHONPATH директорию проекта
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.drift_monitor import check_drift
from scripts.automl_trainer import train_and_save
from scripts.mlflow_registry import MLflowModelRegistry


# ============================================================================
# КОНФИГУРАЦИЯ DAG
# ============================================================================
DEFAULT_ARGS = {
    'owner': 'ml-engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'depends_on_past': False,
}

# ============================================================================
# ОПРЕДЕЛЕНИЕ DAG
# ============================================================================
dag = DAG(
    dag_id='automl_retraining_pipeline',
    description='ML Pipeline: мониторинг drift → AutoML → MLflow → A/B тест',
    schedule_interval='@daily',  # Запускать ежедневно
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=['ml', 'automl', 'monitoring'],
)


# ============================================================================
# TASK ФУНКЦИИ
# ============================================================================

def task_check_drift(**context) -> str:
    """
    Задача 1: Проверка data drift.

    Вызывает функцию check_drift из scripts/drift_monitor.py и выталкивает
    результат в XCom для следующих задач.

    Возвращает:
        str: 'train_automl' если drift обнаружен, 'skip_training' иначе

    XCom:
        Сохраняет 'drift_detected' (bool) и 'drift_metrics' (dict)
    """
    try:
        import logging
        logger = logging.getLogger(__name__)

        logger.info("=" * 80)
        logger.info("TASK: check_data_drift")
        logger.info("=" * 80)

        # Вызвать функцию проверки drift
        drift_detected, drift_metrics = check_drift()

        # Сохранить результаты в XCom для следующих задач
        context['task_instance'].xcom_push(
            key='drift_detected',
            value=drift_detected
        )
        context['task_instance'].xcom_push(
            key='drift_metrics',
            value=drift_metrics
        )

        logger.info(f"Drift detected: {drift_detected}")
        logger.info(f"Drift metrics: {drift_metrics}")

        # Вернуть ID следующей задачи для BranchOperator
        if drift_detected:
            logger.info("→ Переход к task_train_automl")
            return 'train_automl'
        else:
            logger.info("→ Пропуск обучения, переход к завершению")
            return 'notify_complete'

    except Exception as e:
        logger.error(f"Ошибка в check_data_drift: {str(e)}", exc_info=True)
        raise


def task_train_automl(**context) -> Dict[str, Any]:
    """
    Задача 2: Обучение модели с помощью PyCaret AutoML (только если обнаружен drift).

    Вызывает функцию train_and_save из scripts/automl_trainer.py, которая:
    1. Загружает reference датасет
    2. Инициализирует PyCaret с log_experiment=True (автологирование в MLflow)
    3. Запускает compare_models() для поиска лучшей модели
    4. Сохраняет лучшую модель
    5. Возвращает информацию о модели

    XCom:
        Сохраняет 'best_model_info' (dict) с информацией о обученной модели
    """
    try:
        import logging
        logger = logging.getLogger(__name__)

        logger.info("=" * 80)
        logger.info("TASK: train_automl")
        logger.info("=" * 80)

        # Получить информацию о drift из предыдущей задачи
        drift_detected = context['task_instance'].xcom_pull(
            key='drift_detected',
            task_ids='check_drift'
        )

        logger.info(f"Drift detected в предыдущей задаче: {drift_detected}")

        # Вызвать функцию обучения
        best_model_info = train_and_save()

        # Сохранить результаты в XCom
        context['task_instance'].xcom_push(
            key='best_model_info',
            value=best_model_info
        )

        logger.info(f"AutoML завершено: {best_model_info}")
        logger.info("→ Переход к task_register_model")

        return best_model_info

    except Exception as e:
        logger.error(f"Ошибка в train_automl: {str(e)}", exc_info=True)
        raise


def task_register_model(**context) -> Dict[str, Any]:
    """
    Задача 3: Регистрация обученной модели в MLflow Model Registry.

    Вызывает методы MLflowModelRegistry для:
    1. Регистрации модели в Model Registry
    2. Установки стадии "Staging" для тестирования

    Новая модель будет доступна в Flask приложении для A/B теста.

    XCom:
        Получает 'best_model_info' из task_train_automl
        Сохраняет 'registered_model_info' (dict)
    """
    try:
        import logging
        logger = logging.getLogger(__name__)

        logger.info("=" * 80)
        logger.info("TASK: register_model")
        logger.info("=" * 80)

        # Получить информацию о лучшей модели из предыдущей задачи
        best_model_info = context['task_instance'].xcom_pull(
            key='best_model_info',
            task_ids='train_automl'
        )

        logger.info(f"Получена информация о модели: {best_model_info}")

        # Инициализировать Registry
        registry = MLflowModelRegistry()

        # Получить ID последнего экспперимента (из MLflow)
        # Примечание: в production нужно передавать run_id явно
        import mlflow
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else 'unknown'

        logger.info(f"Run ID: {run_id}")

        # Зарегистрировать модель
        registered_info = registry.register_new_model(
            run_id=run_id,
            artifact_path='best_automl_model'
        )

        # Сохранить результаты в XCom
        context['task_instance'].xcom_push(
            key='registered_model_info',
            value=registered_info
        )

        logger.info(f"Модель зарегистрирована: {registered_info}")
        logger.info("→ Переход к task_notify_complete")

        return registered_info

    except Exception as e:
        logger.error(f"Ошибка в register_model: {str(e)}", exc_info=True)
        raise


def task_notify_complete(**context) -> None:
    """
    Задача 4: Логирование завершения pipeline.

    Эта задача является финальной и логирует статус завершения pipeline.
    В production это может быть отправка уведомления, логирование метрик и т.д.

    Получает результаты из XCom всех предыдущих задач и логирует их.
    """
    try:
        import logging
        logger = logging.getLogger(__name__)

        logger.info("=" * 80)
        logger.info("TASK: notify_complete")
        logger.info("=" * 80)

        # Получить результаты из предыдущих задач
        drift_detected = context['task_instance'].xcom_pull(
            key='drift_detected',
            task_ids='check_drift'
        )

        # Попытаться получить информацию о модели (если обучение произошло)
        best_model_info = context['task_instance'].xcom_pull(
            key='best_model_info',
            task_ids='train_automl',
            default=None
        )

        registered_model_info = context['task_instance'].xcom_pull(
            key='registered_model_info',
            task_ids='register_model',
            default=None
        )

        # ================================================================
        # Логирование итогов
        # ================================================================
        logger.info("\n" + "=" * 80)
        logger.info("ИТОГИ PIPELINE ВЫПОЛНЕНИЯ")
        logger.info("=" * 80)

        logger.info(f"Дата: {datetime.now().isoformat()}")
        logger.info(f"DAG ID: {context['dag'].dag_id}")
        logger.info(f"Task ID: {context['task'].task_id}")

        logger.info(f"\nШаг 1 - Проверка drift:")
        logger.info(f"  Обнаружен drift: {drift_detected}")

        if best_model_info:
            logger.info(f"\nШаг 2 - AutoML обучение:")
            logger.info(f"  Модель: {best_model_info.get('model_name')}")
            logger.info(f"  Тип: {best_model_info.get('model_type')}")
            logger.info(f"  Сохранена: {best_model_info.get('saved_model_path')}")
        else:
            logger.info(f"\nШаг 2 - AutoML обучение: ПРОПУЩЕНО (нет drift)")

        if registered_model_info:
            logger.info(f"\nШаг 3 - Регистрация в MLflow:")
            logger.info(f"  Версия: {registered_model_info.get('model_version')}")
            logger.info(f"  Стадия: {registered_model_info.get('stage')}")
            logger.info(f"  URI: {registered_model_info.get('model_uri')}")
        else:
            logger.info(f"\nШаг 3 - Регистрация в MLflow: ПРОПУЩЕНА (нет нового обучения)")

        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE ЗАВЕРШЁН УСПЕШНО")
        logger.info("=" * 80)

        # В production окружении здесь можно:
        # - Отправить Slack уведомление
        # - Логировать метрики в мониторинг систему
        # - Обновить дашборд
        # - Запустить A/B тест аналитику

    except Exception as e:
        logger.error(f"Ошибка в notify_complete: {str(e)}", exc_info=True)
        raise


# ============================================================================
# ОПРЕДЕЛЕНИЕ ЗАДАЧ В DAG
# ============================================================================

# Задача 1: Проверка drift
check_drift_task = BranchPythonOperator(
    task_id='check_drift',
    python_callable=task_check_drift,
    provide_context=True,
    dag=dag,
)

# Задача 2: Обучение AutoML (условная)
train_automl_task = PythonOperator(
    task_id='train_automl',
    python_callable=task_train_automl,
    provide_context=True,
    dag=dag,
)

# Задача 3: Регистрация модели (условная)
register_model_task = PythonOperator(
    task_id='register_model',
    python_callable=task_register_model,
    provide_context=True,
    dag=dag,
)

# Dummy задача для пропуска обучения если drift не обнаружен
skip_task = DummyOperator(
    task_id='skip_training',
    dag=dag,
)

# Задача 4: Завершение
notify_task = PythonOperator(
    task_id='notify_complete',
    python_callable=task_notify_complete,
    provide_context=True,
    trigger_rule='none_failed_or_skipped',  # Выполнить независимо от предыдущих
    dag=dag,
)

# ============================================================================
# ОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ МЕЖДУ ЗАДАЧАМИ
# ============================================================================

# check_drift выбирает следующую задачу:
# - Если drift обнаружен → train_automl
# - Если drift не обнаружен → skip_training
check_drift_task >> [train_automl_task, skip_task]

# Если обучение произошло → регистрация модели
train_automl_task >> register_model_task

# Обе ветки (обучение и пропуск) ведут к завершению
[register_model_task, skip_task] >> notify_task

# ============================================================================
# ДОКУМЕНТАЦИЯ DAG
# ============================================================================
"""
DAG ДОКУМЕНТАЦИЯ:

Назначение:
    Автоматизированный ML pipeline для переобучения моделей при обнаружении
    data drift с регистрацией в MLflow и подготовкой к A/B тесту.

Расписание:
    @daily — один раз в сутки (можно изменить в schedule_interval)

Этапы выполнения:

1. check_data_drift (Обязательная)
   - Проверяет наличие data drift (PSI, KS-тест)
   - Пороговое значение: 0.2 (настраивается в .env)
   - Если drift обнаружен → переход к обучению
   - Если drift не обнаружен → пропуск обучения

2. train_automl (Условная — только при drift)
   - PyCaret AutoML для выбора лучшей модели
   - Автологирование экспериментов в MLflow (log_experiment=True)
   - Попробует: LR, KNN, NB, DT, RF, SVM, XGB, LightGBM и др.
   - Сохранит лучшую модель локально

3. register_model (Условная — только после обучения)
   - Регистрация модели в MLflow Model Registry
   - Установка стадии "Staging" для A/B теста
   - После успешного теста может быть переведена в "Production"

4. notify_complete (Обязательная)
   - Логирование результатов выполнения
   - В production — отправка уведомлений в систему мониторинга

Переменные окружения (.env):
    DATA_PATH=data/reference.csv              # Путь к обучающим данным
    REFERENCE_DATA_PATH=data/reference.csv    # Reference датасет для drift
    CURRENT_DATA_PATH=data/current.csv        # Current датасет для drift
    TARGET_COLUMN=Survived                    # Целевая переменная
    DRIFT_THRESHOLD=0.2                       # Порог PSI для drift
    MLFLOW_TRACKING_URI=./mlruns              # MLflow tracking
    MODEL_NAME=automl-best-model              # Имя модели в Registry

Примеры использования:

# Запустить DAG вручную
airflow dags trigger automl_retraining_pipeline

# Смотреть логи
airflow logs automl_retraining_pipeline check_drift -v

# Проверить статус
airflow dags list-runs automl_retraining_pipeline

Интеграция с Flask приложением:

После регистрации модели в MLflow Registry, Flask приложение (app.py)
автоматически загружает Staging версию для A/B теста:

- Production модель: текущая модель в production
- Staging модель: новая обученная модель для тестирования

Метрики A/B теста логируются в logs/ab_test_log.csv и анализируются
скриптом scripts/analyze_results.py.

Если A/B тест успешен:
    → Staging модель переводится в Production (mlflow_registry.promote_to_production())
    → Старая Production модель архивируется
"""
