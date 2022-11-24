"""
В модуле описан класс подгружающий конфигурацию БД.
"""
import os
from dotenv import load_dotenv


# pylint: disable=too-few-public-methods
class ConfigLoader:
    """
    Класс позволяет загрузить конфигурацию для БД
    """

    def __init__(self):
        load_dotenv()
        self.minio_host = os.environ.get('API_HOST')
        self.minio_user = os.environ.get('API_USER')
        self.minio_pass = os.environ.get('API_KEY')
