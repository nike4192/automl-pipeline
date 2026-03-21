"""
Flask A/B тест маршрутизатор для сравнения Production и Staging моделей.

Этап 4 ML-процесса: реализация Flask API-роутера для A/B теста с логированием
всех запросов, версии модели и предсказаний для последующего анализа результатов.

Архитектура:
- GET /predict?features=... — основной маршрут для предсказаний
- A/B разделение: случайное распределение или по user_id % 2
- Логирование: timestamp, model_version, model_stage, input_features, prediction, response_time_ms
- POST /config — динамическое изменение доли трафика
"""

import logging
import os
import json
import time
import csv
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import mlflow
from pycaret.classification import load_model as pycaret_load_model, predict_model as pycaret_predict_model
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ FLASK
# ============================================================================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ============================================================================
# КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================================
# Модели загруженные из MLflow Registry
PRODUCTION_MODEL = None
STAGING_MODEL = None

# Конфиг A/B теста
AB_TRAFFIC_SPLIT = float(os.getenv('AB_TRAFFIC_SPLIT', '0.5'))  # По умолчанию 50/50
AB_STRATEGY = os.getenv('AB_STRATEGY', 'random')  # 'random' или 'user_id'

# Путь к логам
LOG_DIR = os.getenv('LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'ab_test_log.csv')

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ ЛОГИРОВАНИЯ
# ============================================================================
os.makedirs(LOG_DIR, exist_ok=True)

# Инициализировать CSV логи с заголовками
if not os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'model_stage',
                'model_version',
                'input_features',
                'prediction',
                'response_time_ms',
                'user_id'
            ])
        logger.info(f"Создан новый лог файл: {LOG_FILE}")
    except Exception as e:
        logger.error(f"Ошибка при создании лог файла: {str(e)}")


def load_models() -> bool:
    """
    Загрузить Production и Staging модели из MLflow Registry.

    Использует mlflow.sklearn.load_model для загрузки PyCaret pipeline,
    а pycaret.predict_model для предсказаний (корректная обработка NaN).

    Returns:
        bool: True если обе модели загружены, False иначе
    """
    global PRODUCTION_MODEL, STAGING_MODEL

    try:
        logger.info("Загрузка моделей из MLflow Registry...")

        model_name = os.getenv('MODEL_NAME', 'automl-best-model')
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', './mlruns')

        mlflow.set_tracking_uri(tracking_uri)

        # ================================================================
        # 1. Загрузить Production модель (sklearn pipeline через MLflow)
        # ================================================================
        try:
            production_uri = f'models:/{model_name}/Production'
            PRODUCTION_MODEL = mlflow.sklearn.load_model(production_uri)
            logger.info(f"Production модель загружена: {production_uri}")
        except Exception as e:
            logger.warning(f"Production модель не найдена: {str(e)}")
            PRODUCTION_MODEL = None

        # ================================================================
        # 2. Загрузить Staging модель (sklearn pipeline через MLflow)
        # ================================================================
        try:
            staging_uri = f'models:/{model_name}/Staging'
            STAGING_MODEL = mlflow.sklearn.load_model(staging_uri)
            logger.info(f"Staging модель загружена: {staging_uri}")
        except Exception as e:
            logger.warning(f"Staging модель не найдена: {str(e)}")
            STAGING_MODEL = None

        # Проверить что хотя бы одна модель загружена
        if PRODUCTION_MODEL is None and STAGING_MODEL is None:
            logger.error("Критическая ошибка: обе модели не загружены!")
            return False

        logger.info("Модели успешно загружены")
        return True

    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке моделей: {str(e)}", exc_info=True)
        return False


def determine_model_variant(user_id: Optional[str] = None) -> Tuple[str, Any, str]:
    """
    Определить, какой вариант модели (Production или Staging) использовать.

    A/B разделение: случайное или по user_id % 2 (в зависимости от AB_STRATEGY)

    Логика:
    - random: random() < AB_TRAFFIC_SPLIT → Production (A), иначе Staging (B)
    - user_id: user_id % 2 == 0 → Production (A), иначе Staging (B)

    Args:
        user_id: ID пользователя (опционально, используется для user_id стратегии)

    Returns:
        Tuple:
        - str: 'Production' или 'Staging'
        - model: загруженная модель
        - str: версия модели
    """
    try:
        # ================================================================
        # 1. Выбрать стратегию разделения трафика
        # ================================================================
        if AB_STRATEGY == 'user_id' and user_id:
            # Детерминированное разделение по user_id
            # user_id % 2 == 0 → Production, иначе Staging
            use_production = int(user_id) % 2 == 0
            logger.debug(f"A/B стратегия: user_id ({user_id}), use_production={use_production}")
        else:
            # Случайное разделение на основе AB_TRAFFIC_SPLIT
            use_production = random.random() < AB_TRAFFIC_SPLIT
            logger.debug(f"A/B стратегия: random, AB_TRAFFIC_SPLIT={AB_TRAFFIC_SPLIT}, use_production={use_production}")

        # ================================================================
        # 2. Выбрать модель и получить версию
        # ================================================================
        if use_production and PRODUCTION_MODEL is not None:
            stage = 'Production'
            model = PRODUCTION_MODEL
            version = 'prod_v1'  # Можно расширить для получения реального номера версии
        elif STAGING_MODEL is not None:
            stage = 'Staging'
            model = STAGING_MODEL
            version = 'staging_v1'
        elif PRODUCTION_MODEL is not None:
            # Fallback на Production если Staging недоступна
            logger.warning("Staging модель недоступна, используется Production")
            stage = 'Production'
            model = PRODUCTION_MODEL
            version = 'prod_v1'
        else:
            raise RuntimeError("Обе модели недоступны!")

        return stage, model, version

    except Exception as e:
        logger.error(f"Ошибка при определении варианта модели: {str(e)}", exc_info=True)
        raise


def log_prediction(
    timestamp: str,
    model_stage: str,
    model_version: str,
    input_features: str,
    prediction: Any,
    response_time_ms: float,
    user_id: Optional[str] = None
) -> None:
    """
    Логировать результат предсказания в CSV файл.

    Формат логов:
    timestamp, model_stage, model_version, input_features, prediction, response_time_ms, user_id

    Args:
        timestamp: ISO 8601 временная метка
        model_stage: 'Production' или 'Staging'
        model_version: версия модели
        input_features: JSON строка с признаками (для отладки)
        prediction: результат предсказания
        response_time_ms: время выполнения в миллисекундах
        user_id: ID пользователя (опционально)
    """
    try:
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                model_stage,
                model_version,
                input_features,
                prediction,
                response_time_ms,
                user_id or 'unknown'
            ])
    except Exception as e:
        logger.error(f"Ошибка при логировании предсказания: {str(e)}")


# ============================================================================
# FLASK МАРШРУТЫ
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    GET /health — проверка здоровья приложения.

    Returns:
        JSON с информацией о статусе приложения и загруженных моделях
    """
    try:
        status = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'models': {
                'production': PRODUCTION_MODEL is not None,
                'staging': STAGING_MODEL is not None,
            },
            'ab_config': {
                'traffic_split': AB_TRAFFIC_SPLIT,
                'strategy': AB_STRATEGY
            }
        }

        logger.info(f"Health check: {status}")
        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Ошибка в health check: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    GET/POST /predict — основной маршрут для предсказаний с A/B разделением.

    Query параметры (GET):
        features: JSON строка с признаками (например: '{"Age": 30, "Sex": "male"}')
        user_id: ID пользователя (опционально, для user_id стратегии)
        true_label: истинное значение целевой переменной (опционально, для обучения)

    JSON (POST):
        {
            "features": {...},
            "user_id": "user_123",
            "true_label": 1
        }

    Returns:
        JSON с результатом предсказания и метаданными модели:
        {
            "prediction": ...,
            "model_stage": "Production" | "Staging",
            "model_version": "...",
            "response_time_ms": float
        }
    """
    start_time = time.time()

    try:
        # ================================================================
        # 1. Извлечь входные данные
        # ================================================================
        if request.method == 'POST':
            data = request.get_json()
            features = data.get('features')
            user_id = data.get('user_id')
            true_label = data.get('true_label')
        else:  # GET
            features_str = request.args.get('features')
            user_id = request.args.get('user_id')
            true_label = request.args.get('true_label')

            # Парсировать JSON строку признаков
            if features_str:
                features = json.loads(features_str)
            else:
                features = None

        # ================================================================
        # 2. Валидация входных данных
        # ================================================================
        if features is None:
            return jsonify({
                'error': 'Missing features parameter',
                'example': 'GET /predict?features={\"Age\":30,\"Sex\":\"male\"}'
            }), 400

        # ================================================================
        # 3. Определить вариант модели (A/B разделение)
        # ================================================================
        model_stage, model, model_version = determine_model_variant(user_id)
        logger.info(f"A/B выбор: {model_stage} (user_id={user_id})")

        # ================================================================
        # 4. Сделать предсказание
        # ================================================================
        # Подготовить данные для модели (может потребоваться преобразование)
        # PyCaret модель ожидает DataFrame со всеми колонками из обучающего набора
        import pandas as pd
        # Дополнить недостающие колонки значениями по умолчанию
        all_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
                       'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        for col in all_columns:
            if col not in features:
                features[col] = None
        features_df = pd.DataFrame([features])

        # Используем pycaret predict_model для корректной обработки pipeline
        result_df = pycaret_predict_model(model, data=features_df)
        prediction_value = int(result_df['prediction_label'].iloc[0])

        # ================================================================
        # 5. Вычислить время выполнения
        # ================================================================
        response_time_ms = (time.time() - start_time) * 1000

        # ================================================================
        # 6. Логировать результат
        # ================================================================
        timestamp = datetime.now().isoformat()
        features_json = json.dumps(features, ensure_ascii=False)

        log_prediction(
            timestamp=timestamp,
            model_stage=model_stage,
            model_version=model_version,
            input_features=features_json,
            prediction=prediction_value,
            response_time_ms=response_time_ms,
            user_id=user_id
        )

        logger.info(
            f"Предсказание: model={model_stage}, "
            f"prediction={prediction_value}, "
            f"time={response_time_ms:.2f}ms"
        )

        # ================================================================
        # 7. Вернуть результат
        # ================================================================
        result = {
            'prediction': float(prediction_value) if isinstance(prediction_value, (int, float)) else prediction_value,
            'model_stage': model_stage,
            'model_version': model_version,
            'response_time_ms': round(response_time_ms, 2)
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/config', methods=['POST'])
def update_ab_config():
    """
    POST /config — динамическое изменение конфигурации A/B теста.

    JSON параметры:
        {
            "traffic_split": 0.7,      # Доля трафика для Production (0.0-1.0)
            "strategy": "random"       # "random" или "user_id"
        }

    Returns:
        JSON с новыми параметрами конфигурации
    """
    global AB_TRAFFIC_SPLIT, AB_STRATEGY

    try:
        data = request.get_json()

        # ================================================================
        # 1. Обновить traffic_split если указан
        # ================================================================
        if 'traffic_split' in data:
            new_split = float(data['traffic_split'])
            if not (0.0 <= new_split <= 1.0):
                return jsonify({
                    'error': 'traffic_split must be between 0.0 and 1.0'
                }), 400

            AB_TRAFFIC_SPLIT = new_split
            logger.info(f"Traffic split обновлён: {AB_TRAFFIC_SPLIT}")

        # ================================================================
        # 2. Обновить стратегию если указана
        # ================================================================
        if 'strategy' in data:
            new_strategy = data['strategy']
            if new_strategy not in ['random', 'user_id']:
                return jsonify({
                    'error': 'strategy must be "random" or "user_id"'
                }), 400

            AB_STRATEGY = new_strategy
            logger.info(f"A/B стратегия обновлена: {AB_STRATEGY}")

        # ================================================================
        # 3. Вернуть обновлённую конфигурацию
        # ================================================================
        config = {
            'traffic_split': AB_TRAFFIC_SPLIT,
            'strategy': AB_STRATEGY,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Новая конфигурация: {config}")
        return jsonify(config), 200

    except Exception as e:
        logger.error(f"Ошибка при обновлении конфига: {str(e)}")
        return jsonify({
            'error': 'Config update failed',
            'message': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    GET /metrics — получить сводку собранных метрик из логов.

    Returns:
        JSON со сводкой метрик по моделям
    """
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({'message': 'No metrics collected yet'}), 200

        # Загрузить логи и вычислить базовые метрики
        import pandas as pd
        df = pd.read_csv(LOG_FILE)

        if df.empty:
            return jsonify({'message': 'No metrics collected yet'}), 200

        metrics = {
            'total_predictions': len(df),
            'total_models': {
                'production': len(df[df['model_stage'] == 'Production']),
                'staging': len(df[df['model_stage'] == 'Staging'])
            },
            'avg_response_time_ms': float(df['response_time_ms'].mean()),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Metrics retrieved: {metrics}")
        return jsonify(metrics), 200

    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {str(e)}")
        return jsonify({
            'error': 'Metrics retrieval failed',
            'message': str(e)
        }), 500


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ И ЗАПУСК
# ============================================================================

@app.before_request
def before_request():
    """Хук до каждого запроса."""
    # Загрузить модели если еще не загружены
    if PRODUCTION_MODEL is None and STAGING_MODEL is None:
        if not load_models():
            logger.error("Не удалось загрузить модели")


if __name__ == '__main__':
    """
    Точка входа для запуска Flask приложения.
    """
    logger.info("=" * 80)
    logger.info("ЗАПУСК FLASK A/B ТЕСТ ПРИЛОЖЕНИЯ")
    logger.info("=" * 80)

    # Попытаться загрузить модели при запуске
    if not load_models():
        logger.warning("Не удалось загрузить модели при запуске")
        logger.warning("Приложение запустится, но /predict вернёт ошибку")

    # Запустить Flask приложение
    port = int(os.getenv('FLASK_PORT', '5000'))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    logger.info(f"Запуск на {host}:{port}, debug={debug}")

    app.run(host=host, port=port, debug=debug)
