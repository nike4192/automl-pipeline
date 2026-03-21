# AutoML Pipeline: от мониторинга до A/B теста

## Структура заданий

| # | Задание | Файлы |
|---|---------|-------|
| 1 | Model Card + BPMN использования модели | [`MODEL_CARD.md`](MODEL_CARD.md), [`bpmn/model_usage.bpmn`](bpmn/model_usage.bpmn) |
| 2 | BPMN переобучения модели (Human in the Loop) | [`bpmn/model_retraining.bpmn`](bpmn/model_retraining.bpmn) |
| 3 | Детекция Data Drift (Evidently + PSI/KS) | [`scripts/drift_detection_evidently.py`](scripts/drift_detection_evidently.py), [`scripts/drift_monitor.py`](scripts/drift_monitor.py) |
| 4 | Итоговый проект: ML-пайплайн от мониторинга до A/B теста | Весь репозиторий |

> BPMN-диаграммы можно открыть на [bpmn.io](https://bpmn.io/) (File → Open)

## Описание проекта

Production-готовый ML-пайплайн, реализующий полный цикл автоматического переобучения модели:

1. **Мониторинг Data Drift** — обнаружение сдвига данных (PSI, KS-тест, сравнение средних)
2. **AutoML с PyCaret** — автоматический подбор лучшей модели из 15+ алгоритмов
3. **MLflow Model Registry** — версионирование и управление жизненным циклом моделей
4. **A/B тестирование через Flask** — роутер с динамическим распределением трафика
5. **Анализ результатов** — статистические тесты, ML и бизнес-метрики

Датасет: **Titanic** (фокус на инфраструктуре, а не на данных).

## Структура репозитория

```
automl-pipeline/
├── app.py                          # Flask A/B тест роутер
├── requirements.txt                # Зависимости
├── .env                            # Конфигурация (не коммитить в git!)
├── .env.example                    # Шаблон конфигурации
├── README.md                       # Документация
├── dags/
│   └── ml_pipeline_dag.py          # Airflow DAG — оркестрация всего пайплайна
├── scripts/
│   ├── __init__.py
│   ├── data_utils.py               # Загрузка данных, генерация drift
│   ├── drift_monitor.py            # Мониторинг data drift (PSI, KS, mean shift)
│   ├── automl_trainer.py           # PyCaret AutoML — обучение и выбор модели
│   ├── mlflow_registry.py          # MLflow Model Registry — регистрация моделей
│   └── analyze_results.py          # Анализ A/B теста, статистика, визуализации
├── configs/
│   └── airflow_variables.json      # Переменные Airflow
├── data/                           # Данные (генерируются автоматически)
│   ├── reference.csv
│   └── current.csv
├── models/                         # Сохранённые модели
└── logs/                           # Логи и отчёты
    ├── ab_test_log.csv
    ├── drift_history.jsonl
    └── ab_test_report.txt
```

## Быстрый старт

### 1. Установка зависимостей

```bash
# Через Poetry (рекомендуется)
poetry install

# Или через pip
pip install -r requirements.txt
```

### 2. Подготовка данных

```bash
# Генерация reference и current датасетов с синтетическим drift
python -m scripts.data_utils
```

### 3. Проверка Data Drift

```bash
python -m scripts.drift_monitor
# Вывод: Drift detected: True, Max PSI: 1.2345
```

### 4. Обучение модели (AutoML)

```bash
# Запуск MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 &

# Обучение с PyCaret
python -m scripts.automl_trainer
```

### 5. Регистрация модели

```bash
python -c "
from scripts.mlflow_registry import MlflowModelRegistry
registry = MlflowModelRegistry()
registry.register_new_model()
"
```

### 6. Запуск A/B теста

```bash
# Запуск Flask-сервера
python app.py

# Отправка тестовых запросов
curl 'http://localhost:8080/predict?Pclass=3&Sex=male&Age=22&SibSp=1&Parch=0&Fare=7.25'
curl 'http://localhost:8080/health'
curl 'http://localhost:8080/metrics'

# Изменение доли трафика
curl -X POST 'http://localhost:8080/config' \
  -H 'Content-Type: application/json' \
  -d '{"traffic_split": 0.5}'
```

### 7. Анализ результатов

```bash
python -m scripts.analyze_results
# Отчёт сохраняется в logs/ab_test_report.txt
# Графики — в logs/*.png
```

### 8. Запуск через Airflow (оркестрация)

```bash
# Скопировать DAG в Airflow
cp dags/ml_pipeline_dag.py $AIRFLOW_HOME/dags/

# Запустить Airflow
airflow standalone
# DAG "automl_retraining_pipeline" будет запускаться ежедневно
```

## Этапы работы (подробно)

### Этап 1. Мониторинг Data Drift

Скрипт `scripts/drift_monitor.py` реализует три метода обнаружения drift:

| Метод | Описание | Порог |
|-------|----------|-------|
| **PSI** (Population Stability Index) | Сравнение распределений reference vs current | PSI > 0.2 |
| **KS-тест** (Колмогоров-Смирнов) | Статистический тест на одинаковость распределений | p-value < 0.05 |
| **Сдвиг среднего** | Проверка |mean_cur - mean_ref| > k * std_ref | k = 1.0 |

Решение о drift принимается на основе PSI — если хотя бы один признак превышает порог.

### Этап 2. AutoML с PyCaret

Скрипт `scripts/automl_trainer.py`:
- `setup()` — инициализация с автологированием в MLflow
- `compare_models()` — сравнение 15+ алгоритмов (LR, RF, XGBoost, LightGBM, ...)
- `save_model()` — сохранение лучшей модели
- Без ручных шагов: setup → compare → save

### Этап 3. MLflow Model Registry

Скрипт `scripts/mlflow_registry.py`:
- Регистрация модели в Model Registry
- Переходы между стадиями: `None → Staging → Production`
- Загрузка Production/Staging моделей для A/B теста

### Этап 4. A/B тест через Flask

Файл `app.py` — REST API:
- **Стратегии распределения:** случайное (random) или по user_id % 2
- **Динамический сплит:** можно менять 70/30, 50/50 и т.д. через POST /config
- **Логирование:** каждый запрос записывается в CSV с timestamp, версией модели, prediction

### Этап 5. Анализ результатов

Скрипт `scripts/analyze_results.py`:
- **ML-метрики:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Бизнес-метрики:** среднее время ответа, распределение предсказаний
- **Статистика:** T-тест, Chi-square, p-value
- **Визуализации:** графики сравнения моделей A и B

## Airflow DAG

```
check_data_drift
    │
    ├── [drift обнаружен] → train_automl → register_model → notify_complete
    │
    └── [drift не обнаружен] → skip_training → notify_complete
```

DAG запускается ежедневно. При обнаружении drift автоматически:
1. Обучает новую модель через PyCaret
2. Регистрирует её в MLflow как Staging
3. Уведомляет о готовности к A/B тестированию

## Конфигурация

Все параметры задаются через `.env`:

| Переменная | Описание | По умолчанию |
|------------|----------|-------------|
| `DRIFT_THRESHOLD_PSI` | Порог PSI для drift | 0.2 |
| `TARGET_COLUMN` | Целевая переменная | Survived |
| `MLFLOW_TRACKING_URI` | URI MLflow сервера | http://localhost:5000 |
| `MODEL_NAME` | Имя модели в Registry | automl-best-model |
| `AB_TRAFFIC_SPLIT` | Доля трафика на Production | 0.7 |
| `FLASK_PORT` | Порт Flask-сервера | 8080 |

## Инструменты

- **Apache Airflow** — оркестрация пайплайна
- **PyCaret** — AutoML
- **MLflow** — трекинг экспериментов, Model Registry
- **Flask** — A/B тест роутер
- **scipy** — статистические тесты (KS, t-test, chi-square)
- **matplotlib** — визуализации
- **python-dotenv** — конфигурация через .env
