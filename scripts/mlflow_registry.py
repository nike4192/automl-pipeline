"""
MLflow Model Registry — регистрация и управление версиями моделей.

Этап 3 ML-процесса: регистрация обученной модели в MLflow Model Registry
и управление её жизненным циклом (None → Staging → Production).
"""

import logging
import os
from typing import Dict, Any, Optional

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowModelRegistry:
    """
    Менеджер для работы с MLflow Model Registry.

    Позволяет регистрировать модели, переводить их через стадии
    (Staging → Production) и загружать версии для предсказаний.
    """

    def __init__(self):
        """Инициализация клиента MLflow Registry."""
        try:
            # Получить MLflow Tracking URI из .env (или использовать локальный)
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
            mlflow.set_tracking_uri(tracking_uri)

            self.client = MlflowClient(tracking_uri=tracking_uri)
            self.model_name = os.getenv('MODEL_NAME', 'automl-best-model')

            logger.info(f"MLflow Registry инициализирован")
            logger.info(f"Tracking URI: {tracking_uri}")
            logger.info(f"Имя модели: {self.model_name}")

        except Exception as e:
            logger.error(f"Ошибка при инициализации MLflow Registry: {str(e)}")
            raise

    def register_new_model(self, run_id: str, artifact_path: str = 'best_automl_model') -> Dict[str, Any]:
        """
        Зарегистрировать новую модель в MLflow Model Registry и установить стадию "Staging".

        Эта функция вызывается после успешного обучения модели в PyCaret.
        Она регистрирует модель в глобальном реестре и переводит её в стадию Staging
        для последующего тестирования через A/B тест перед попаданием в Production.

        Args:
            run_id: ID эксперимента MLflow (из метаданных обучения)
            artifact_path: Путь артефакта модели внутри эксперимента
                          (по умолчанию совпадает с именем модели в PyCaret)

        Returns:
            Dict с информацией о зарегистрированной модели:
            {
                'model_uri': str,          # URI модели (вида models:/name/stage)
                'model_version': int,      # Номер версии
                'stage': str,              # Текущая стадия ("Staging")
                'status': str              # Статус регистрации
            }

        Raises:
            Exception: Если регистрация не удалась
        """
        try:
            logger.info(f"Регистрация модели '{self.model_name}' из run_id={run_id}")

            # Полный URI артефакта в формате MLflow
            model_uri = f'runs:/{run_id}/{artifact_path}'

            # ================================================================
            # 1. Регистрация модели в Model Registry
            # ================================================================
            # mlflow.register_model() создаёт новую версию модели в реестре
            # Первая версия получает номер 1, вторая — 2 и т.д.
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name
            )

            logger.info(
                f"Модель успешно зарегистрирована: "
                f"{self.model_name}@v{model_version.version}"
            )

            # ================================================================
            # 2. Переход в стадию "Staging"
            # ================================================================
            # Стадия Staging = модель готова к тестированию (A/B тест)
            # После успешного A/B теста модель переводится в Production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage='Staging'
            )

            logger.info(
                f"Модель {self.model_name}@v{model_version.version} "
                f"переведена в стадию 'Staging'"
            )

            result = {
                'model_uri': model_uri,
                'model_version': model_version.version,
                'stage': 'Staging',
                'status': 'registered'
            }

            logger.info(f"Результат регистрации: {result}")
            return result

        except Exception as e:
            logger.error(f"Ошибка при регистрации модели: {str(e)}", exc_info=True)
            raise

    def promote_to_production(self, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Перевести модель из стадии "Staging" в "Production".

        Эта функция вызывается после успешного A/B теста.
        Новая модель в Production становится основной для обслуживания запросов.
        Старая Production модель переводится в "Archived".

        Args:
            version: Номер версии модели (опционально).
                    Если не указана, используется последняя Staging версия.

        Returns:
            Dict с информацией о переведённой модели:
            {
                'model_name': str,
                'version': int,
                'new_stage': str,          # "Production"
                'old_version_archived': int | None
            }

        Raises:
            ValueError: Если нет Staging версии для продвижения
            Exception: Если переход не удался
        """
        try:
            # ================================================================
            # 1. Найти текущую Staging версию (если version не указана)
            # ================================================================
            if version is None:
                staging_versions = self.client.get_latest_versions(
                    name=self.model_name,
                    stages=['Staging']
                )

                if not staging_versions:
                    raise ValueError(
                        f"Нет Staging версии модели {self.model_name} "
                        "для продвижения в Production"
                    )

                version = staging_versions[0].version
                logger.info(f"Найдена Staging версия: v{version}")

            # ================================================================
            # 2. Архивировать текущую Production версию (если она есть)
            # ================================================================
            old_production_versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=['Production']
            )

            old_version = None
            if old_production_versions:
                old_version = old_production_versions[0].version
                logger.info(f"Архивирование старой Production версии: v{old_version}")

                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=old_version,
                    stage='Archived'
                )

            # ================================================================
            # 3. Перевести Staging версию в Production
            # ================================================================
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage='Production'
            )

            logger.info(
                f"Модель {self.model_name}@v{version} "
                f"успешно переведена в Production"
            )

            result = {
                'model_name': self.model_name,
                'version': version,
                'new_stage': 'Production',
                'old_version_archived': old_version
            }

            return result

        except Exception as e:
            logger.error(
                f"Ошибка при продвижении модели в Production: {str(e)}",
                exc_info=True
            )
            raise

    def get_production_model(self) -> Dict[str, Any]:
        """
        Получить текущую Production версию модели.

        Returns:
            Dict с метаданными Production модели или None если её нет:
            {
                'model_name': str,
                'version': int,
                'stage': str,              # "Production"
                'uri': str                 # Полный URI для загрузки
            }

        Raises:
            Exception: При ошибках доступа к реестру
        """
        try:
            production_versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=['Production']
            )

            if not production_versions:
                logger.warning(f"Production версия модели {self.model_name} не найдена")
                return None

            version_obj = production_versions[0]

            result = {
                'model_name': self.model_name,
                'version': version_obj.version,
                'stage': 'Production',
                'uri': f'models:/{self.model_name}/Production'
            }

            logger.info(f"Получена Production версия: v{version_obj.version}")
            return result

        except Exception as e:
            logger.error(f"Ошибка при получении Production модели: {str(e)}")
            raise

    def get_staging_model(self) -> Dict[str, Any]:
        """
        Получить текущую Staging версию модели.

        Returns:
            Dict с метаданными Staging модели или None если её нет:
            {
                'model_name': str,
                'version': int,
                'stage': str,              # "Staging"
                'uri': str                 # Полный URI для загрузки
            }

        Raises:
            Exception: При ошибках доступа к реестру
        """
        try:
            staging_versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=['Staging']
            )

            if not staging_versions:
                logger.warning(f"Staging версия модели {self.model_name} не найдена")
                return None

            version_obj = staging_versions[0]

            result = {
                'model_name': self.model_name,
                'version': version_obj.version,
                'stage': 'Staging',
                'uri': f'models:/{self.model_name}/Staging'
            }

            logger.info(f"Получена Staging версия: v{version_obj.version}")
            return result

        except Exception as e:
            logger.error(f"Ошибка при получении Staging модели: {str(e)}")
            raise

    def list_all_versions(self) -> list:
        """
        Получить список всех версий модели со всеми стадиями.

        Returns:
            List с информацией о всех версиях
        """
        try:
            versions = self.client.search_model_versions(
                f"name='{self.model_name}'"
            )

            results = []
            for version in versions:
                results.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'status': version.status,
                    'creation_timestamp': version.creation_timestamp
                })

            logger.info(f"Найдено {len(results)} версий модели {self.model_name}")
            return results

        except Exception as e:
            logger.error(f"Ошибка при получении списка версий: {str(e)}")
            raise


if __name__ == '__main__':
    """
    Пример использования MLflow Registry.

    Этот скрипт демонстрирует основные операции с моделями в реестре.
    """
    try:
        logger.info("=" * 80)
        logger.info("ПРИМЕР: MLflow Model Registry")
        logger.info("=" * 80)

        registry = MLflowModelRegistry()

        # Список всех версий
        logger.info("\nВсе версии модели:")
        versions = registry.list_all_versions()
        for v in versions:
            logger.info(f"  v{v['version']}: {v['stage']} (status={v['status']})")

        # Получить Production модель
        prod = registry.get_production_model()
        if prod:
            logger.info(f"\nProduction: {prod}")

        # Получить Staging модель
        staging = registry.get_staging_model()
        if staging:
            logger.info(f"\nStaging: {staging}")

    except Exception as e:
        logger.error(f"Ошибка в примере: {str(e)}")
