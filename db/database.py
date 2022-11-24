"""
Модуль позволяющий выполнять запросы к базе данных Minio.
"""

import os
import logging
from typing import Optional, List

from minio import Minio
from urllib3.exceptions import MaxRetryError

from db.config import ConfigLoader  # pylint: disable=import-error

log = logging.getLogger(__name__)


class MinioDB:
    """
    Класс, позволяющий выполнять запросы к базе данных (БД) Minio.
    """

    def __init__(self,
                 main_bucket_name: str,
                 tmp_bucket_name: str,
                 logs_path: str,
                 local_download_path: str):
        """
        Функция инициализирует параметры для БД

        Args:
            main_bucket_name (str): Название папки с видео в удаленной БД.
            tmp_bucket_name (str): Название папки в удаленной БД, в которую будут сохраняться фичи видео.
            logs_path (str): Название локальной папки, в которую будет сохраняться лог об актуальном состоянии.
            local_download_path (str):  Путь до локальной папки, в которую будут сохраняться данные из БД.
        """

        config_manager = ConfigLoader()
        self.client = Minio(
            endpoint=config_manager.minio_host,
            access_key=config_manager.minio_user,
            secret_key=config_manager.minio_pass,
            secure=False
        )
        if not MinioDB.check_connection(self):
            log.error("Check your login, password and host for db!")
            raise ConnectionError

        log.info("Successful connection to db.")

        self.main_bucket = main_bucket_name
        self.tmp_bucket = tmp_bucket_name
        self.local_download_path = local_download_path
        self.logs_path = logs_path

        if not self.client.bucket_exists(self.tmp_bucket):
            self.client.make_bucket(self.tmp_bucket)
            log.info("Created temporary bucket.")

    def check_connection(self) -> bool:
        """
        Проверка соединения с БД.
        Returns: True | False - успешность проверки.
        """
        try:
            _ = self.client.list_buckets()
        except MaxRetryError:
            return False
        return True

    def db_get_video_list(self) -> List[str]:
        """
        Returns (List[str]): Список видео из директории с видео в БД.
        """
        generator = self.client.list_objects(self.main_bucket, recursive=True)
        my_list = []
        for obj in generator:
            my_list.append(obj.object_name)
        return my_list

    def db_get_file(self, obj_name_in_db: str, save_path: Optional[str] = None, bucket: str = 'main'):
        """
        Загрузка объекта из БД в локальную директорию.

        Args:
            obj_name_in_db (str): Имя объекта в БД.
            save_path (Optional[str]): Путь до локальной директории, куда подгружать.
            bucket (str): Указание из какой папки БД подгружать (main - основная, tmp - второстепенная).
        """

        filename = os.path.split(obj_name_in_db)[-1]
        save_path = os.path.join(self.local_download_path, filename) if save_path is None else save_path
        if bucket == 'main':
            self.client.fget_object(self.main_bucket, obj_name_in_db, save_path)
        elif bucket == 'tmp':
            self.client.fget_object(self.tmp_bucket, obj_name_in_db, save_path)
        else:
            log.error("Bucket doesn't exist!")
            raise NameError

    def db_put_file(self, file_path: str):
        """
        Подгрузка объекта из локальной директории в БД.
        Args:
            file_path (str): Локальный путь до файла
        """
        filename = os.path.split(file_path)[-1]
        self.client.fput_object(self.tmp_bucket, filename, file_path)
