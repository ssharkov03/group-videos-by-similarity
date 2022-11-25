"""
Модуль содержащий основные содержащий реализацию основных этапов пайплайна решения задачи сравнения и объединения видео.
"""
import os
from typing import List, Optional

import logging
import numpy as np

from video.compare_videos import VideoSimilarityModel  # pylint: disable=import-error
from utils.manipulate_data import load_data, save_data  # pylint: disable=import-error
from db.database import MinioDB  # pylint: disable=import-error
from utils.sort_dict import sort_dict_by_key  # pylint: disable=import-error, ungrouped-imports
from utils.manipulate_data import load_video as read_video  # pylint: disable=import-error
from meta.submeta import init_submeta  # pylint: disable=import-error

log = logging.getLogger(__name__)


# pylint: disable=too-many-arguments, too-many-instance-attributes
class MetaData:
    # pylint: disable=trailing-whitespace
    # pylint: disable=line-too-long
    """
    Класс, содержащий реализацию основных этапов пайплайна решения задачи сравнения и объединения видео,
    а также, позволяющий в процессе сохранять результаты работы в специальную структуру.

    Структура для отслеживания состояния работы (далее мета данные) представляет собой словарь meta_data со следующими полями:
            meta_data['num_videos'] (int): Количество видео в БД.
            meta_data['was_video_downloaded'] (List[bool]): Было ли скачано текущее видео из БД (локальное наличие).
            meta_data['was_video_read'] (List[bool]): Было ли текущее видео считано (переведено в np.ndarray формат).
            meta_data['were_features_extracted'] (List[bool]): Были ли фичи вытянуты из текущего видео и сохранены локально.
            meta_data['were_features_uploaded'] (List[bool]): Были ли вытянутые фичи загружены в БД.
            meta_data['remote_videos_paths'] (List[str]): Пути до каждого видео внутри БД.
            meta_data['local_videos_paths'] (List[Optional[str]]): Локальные пути до каждого видео, после их скачивания из БД.
            meta_data['remote_features_paths'] (List[Optional[str]]): Пути до каждого файла с фичами внутри БД.
            meta_data['local_features_paths'] (List[Optional[str]]): Локальные пути до каждого файла с фичами, после их скачивания из БД или перед их подгрузкой в БД.
            meta_data['videos_duration'] (List[Optional[int]]): Длительность каждого видео в секундах.
            meta_data['videos_filenames'] (List[str]): Названия файлов видео без расширения.
            meta_data['videos_filenames_w_extensions'] (List[str]): Названия файлов видео c расширением.
            meta_data['was_video_with_error'] (List[str]): Были ли ошибки при работе с видео (не считывается в numpy формат например).
            meta_data['num_groups_found']: Число найденных групп.
            meta_data['main_videos_in_groups_indices']: Индексы главных видео среди всех остальных в meta_data['remote_videos_paths']
            meta_data['main_videos_in_groups_videos_paths'] (List[str]): Пути до каждого главного видео внутри БД.
            meta_data['groups_content_video_paths'] (List[List[str]]): Пути в БД до остальных видео в каждой группе.
            meta_data['comparison_submeta'] (List[Optional[dict]]): Отдельная структура для каждого видео, чтобы отслеживать этапы его сравнения с главными видео.
                                                          (более подробно описано в submeta.py)

    В классе описаны следующие этапы пайплайна:
        0) Инициализация необходимых объектов:
            0.0) Мета данные для отслеживания состояния работы.
            0.1) База данных Minio, в которой находятся видео.
            0.2) Модель ViSiL для сравнения видео и вытягивания из них фич.

        1) Вытягивание фич из видео:
            Итерационный процесс (можно распараллелить). Кроме того скорость на текущем этапе зависит от скорости
            интернета и мощности GPU.
            1.0) Текущее видео скачивается из БД в локальную директорию.
            1.1) Видео считывается из локальной директории.
            1.2) Из видео вытягиваются фичи и они сохраняются в локальную директорию.
            1.3) Фичи выгружаются во временное место хранения в БД.
            1.4) Локально удаляются фичи и видео в БД.

        2) Сортировка мета данных:
            Выполняется сортировка по длительностям видео (reversed=True) всех списков (все списки фиксированной длины
            в мета данных отвечают за описание характеристик, связанных с набором видео из базы данных) из словаря
            отвечающего за meta данные с целью применения последующего алгоритма для сравнения видео.
            (т.е. на основе списка длительностей видео сортируем все остальные)

        3) Сравнение видео:
            Сравнение осуществляется последовательно, поэтому процесс вряд ли получится распараллелить (за исключением
            может быть шагов 3.1.0 - 3.1.3). Cкорость на текущем этапе зависит от скорости
            интернета и мощности GPU. Кроме того, о том какие именно параметры влияют непосредственно
            на скорость сравнения двух видео описано в video/compare_videos.py.
            3.0) Фичи текущего видео скачиваются из БД.
                3.1.0) Скачиваются фичи текущего главное[*] видео из БД (главное видео всегда длиннее текущего).
                3.1.1) Сравниваются фичи текущего видео и текущего главного видео.
                3.1.2) Если видео схожи, то текущее видео является подмножеством текущего главного видео. В этом
                       случае текущее видео добавляется в группу соответствующую текущему главному видео. Затем
                       текущее главное видео и текущее видео локально удаляются. Переходим к (3.0) со следующим видео
                       (более коротким, чем текущее) в качестве текущего.

                        Если видео не схожи, то текущее главное видео не является надмножеством текущего видео.
                        В этом случае локально удаляются фичи текущего главного видео и переходим к (3.1.0)
                        со следующим видео (более коротким, чем текущее) в качестве текущего главного.
                3.1.3) Если ни одно главное видео не является надмножеством текущего видео, то текущее видео само
                       становится главным. Далее переходим к (3.0) со следующим видео в качестве текущего.


    [*]: Главным видео называется самое длинное видео в группе. Предполагается, что все остальные видео в группе
         являются частями главного видео.
    [**]: Группой называется множество видео, которые схожи между собой.
    """

    def __init__(self,
                 logs_path: str,
                 meta_logname: str,
                 main_bucket_name: str,
                 tmp_bucket_name: str,
                 path_to_model: str,
                 local_data_save_path: str):
        # pylint: disable=line-too-long
        """
        Реализация нулевого этапа пайплайна.

        Подгрузка модели, базы данных, а также создание структуры для отслеживания состояния работы или
        подгрузка ранее созданной.

        Описание self объектов:

            self.minio_db (MinioDB): Объект соответствующий базе данных MinioDB.
            self.model (ViSiL): Объект модели ViSiL.
            self.model_threshold (float): Пороговое значение для сравнения двух видео.
            self.model_frames_step (int): Шаг по кадрам для более длинного видео (подробнее тут video/compare_videos.py VideoSimilarityModel.compare_videos)
                        [!Чем больше шаг, тем быстрее работает сравнение, однако точность может упасть!]
            self.meta_data (dict): Ранее описанная структура для отслеживания состояния работы.
            self.meta_log_path (str): Локальный путь до файла со структурой.
            self.main_bucket_name (str): Наименование временной директории в БД, где хранятся видео.
            self.tmp_bucket_name (str): Наименование временной директории в БД, куда будут сохраняться фичи из видео.
            self.local_download_path (str): Путь до директории для локального (временного) сохранения данных из БД.

        Args:
            logs_path (str): Путь до директории со структурой для отслеживания состояния работы.
            meta_logname (str): Название файла со структурой для отслеживания состояния работы.
            main_bucket_name (str): Наименование временной директории в БД, где хранятся видео.
            tmp_bucket_name (str): Наименование временной директории в БД, куда будут сохраняться фичи из видео.
            path_to_model (str): Путь до чекпоинта модели ViSiL.
            local_data_save_path (str): Путь до директории для локального (временного) сохранения данных из БД.
        """

        model = VideoSimilarityModel(path_to_model=path_to_model)

        db_obj = MinioDB(main_bucket_name, tmp_bucket_name, logs_path, local_data_save_path)
        self.minio_db: MinioDB = db_obj

        if meta_logname not in os.listdir(logs_path):
            remote_videos_paths: List[str] = db_obj.db_get_video_list()
            meta_data = dict()  # pylint: disable=use-dict-literal
            meta_data['num_videos']: int = len(remote_videos_paths)
            meta_data['was_video_downloaded']: List[bool] = [False for _ in range(meta_data['num_videos'])]
            meta_data['was_video_read']: List[bool] = [False for _ in range(meta_data['num_videos'])]
            meta_data['were_features_extracted']: List[bool] = [False for _ in range(meta_data['num_videos'])]
            meta_data['were_features_uploaded']: List[bool] = [False for _ in range(meta_data['num_videos'])]
            meta_data['remote_videos_paths']: List[str] = remote_videos_paths
            meta_data['local_videos_paths']: List[Optional[str]] = [None for _ in range(meta_data['num_videos'])]
            meta_data['remote_features_paths']: List[Optional[str]] = [None for _ in range(meta_data['num_videos'])]
            meta_data['local_features_paths']: List[Optional[str]] = [None for _ in range(meta_data['num_videos'])]
            meta_data['videos_duration']: List[Optional[int]] = [None for _ in range(meta_data['num_videos'])]
            meta_data['videos_filenames']: List[str] = [os.path.split(video_path)[-1].split('.')[0] for video_path in
                                                        meta_data['remote_videos_paths']]
            meta_data['videos_filenames_w_extensions']: List[str] = [os.path.split(video_path)[-1] for video_path in
                                                                     meta_data['remote_videos_paths']]
            meta_data['was_video_with_error']: List[bool] = [False for _ in range(meta_data['num_videos'])]
            meta_data['num_groups_found']: int = 0
            meta_data['main_videos_in_groups_indices']: List[int] = []
            meta_data['main_videos_in_groups_videos_paths']: List[str] = []
            meta_data['groups_content_video_paths']: List[str] = []
            meta_data['comparison_submeta']: List[Optional[dict]] = [None for _ in range(meta_data['num_videos'])]
        else:
            meta_data = MetaData.load_meta(os.path.join(logs_path, meta_logname))

        self.model = model
        self.model_threshold = 0.75
        self.model_frames_step = 100
        self.meta_data = meta_data
        self.meta_log_path = os.path.join(logs_path, meta_logname)
        self.main_bucket_name = main_bucket_name
        self.tmp_bucket_name = tmp_bucket_name
        self.local_download_path = local_data_save_path

    @staticmethod
    def load_meta(path_to_meta: str) -> dict:
        """
        Функция чтения мета данных.
        Args:
            path_to_meta (str): Путь до файла с мета данными.
        Returns:
            meta_data (dict): Считанный словарь с мета данными.
        """
        meta_data = load_data(path_to_meta)
        return meta_data

    def update_meta(self):
        """
        Функция обновления (сохранения) мета данных.
        """
        save_data(self.meta_data, self.meta_log_path)

    def download_video(self, video_idx: int):
        """
        Функция загрузки видео из БД по индексу в мета данных.
        После завершения работы функции мета данные обновляются.
        Args:
            video_idx (int): Индекс видео из списка в мета данных.
        """
        self.minio_db.db_get_file(str(self.meta_data['remote_videos_paths'][video_idx]))
        local_video_location = os.path.join(str(self.local_download_path),
                                            str(self.meta_data['videos_filenames_w_extensions'][video_idx]))
        self.meta_data['local_videos_paths'][video_idx] = local_video_location
        self.meta_data['was_video_downloaded'][video_idx] = True
        self.update_meta()

    def read_video(self, video_idx: int) -> np.ndarray:
        """
        Функция чтения видео из локальной директории по индексу в мета данных. 
        После завершения работы функции мета данные обновляются.
        Args:
            video_idx (int): Индекс видео из списка в мета данных.
        Returns:
            video_data (np.ndarray): Считанное видео в формате numpy.
        """
        video_data = read_video(self.meta_data['local_videos_paths'][video_idx])
        if not self.meta_data['was_video_read'][video_idx]:
            self.meta_data['videos_duration'][video_idx] = video_data.shape[0]
            self.meta_data['was_video_with_error'][video_idx] = video_data.shape[0] == 0
            self.meta_data['was_video_read'][video_idx] = True
            self.update_meta()
        return video_data

    def extract_features_from_video(self, video_idx: int):
        """
        Функция, для вытягивания фич с видео (по индексу в мета данных) с помощью модели ViSiL для 
        последующего сравнения текущего видео с остальными. После вытягивания фичи сохраняются в локальную
        директорию. После завершения работы функции мета данные обновляются.
        Args:
            video_idx (int): Индекс видео из списка в мета данных.            
        """
        if self.meta_data['was_video_with_error'][video_idx]:
            self.meta_data['were_features_extracted'][video_idx] = True
            self.update_meta()
        else:
            video_data = self.read_video(video_idx)
            video_data = self.model.extract_features(video_data, batch_sz=32)
            features_filename = str(self.meta_data['videos_filenames'][video_idx]) + "_features.pkl"
            local_path_to_features = os.path.join(str(self.local_download_path), features_filename)
            remote_path_to_features = features_filename
            self.meta_data['were_features_extracted'][video_idx] = True
            self.meta_data['local_features_paths'][video_idx] = local_path_to_features
            self.meta_data['remote_features_paths'][video_idx] = remote_path_to_features
            save_data(video_data, local_path_to_features)
            self.update_meta()
            del video_data

    def upload_features(self, video_idx: int):
        """
        Выгрузка локально расположенных фич видео с индексом video_idx в мета данных в базу данных.  
        После выгрузки функция удаляет локально расположенные фичи, а также само видео. 
        После завершения работы функции мета данные обновляются.
        Args:
            video_idx (int): Индекс видео из списка в мета данных.
        """
        if self.meta_data['was_video_with_error'][video_idx]:
            self.meta_data['were_features_uploaded'] = True
            os.remove(str(self.meta_data['local_videos_paths'][video_idx]))
            self.update_meta()
        else:
            # load features in tmp bucket
            self.minio_db.db_put_file(str(self.meta_data['local_features_paths'][video_idx]))
            os.remove(str(self.meta_data['local_features_paths'][video_idx]))
            os.remove(str(self.meta_data['local_videos_paths'][video_idx]))
            self.meta_data['were_features_uploaded'][video_idx] = True
            self.update_meta()

    def preprocessing(self):
        """
        Реализация 1 и 2 этапа пайплайна.
        Вытягивание фич из всех видео и сортировка мета данных.
        После завершения работы функции мета данные обновляются.
        """
        log.info("Реализизация 1 и 2 этапа пайплайна.")
        for video_idx, _ in enumerate(self.meta_data['remote_videos_paths']):
            # pylint: disable=logging-fstring-interpolation
            log.info(f"Обработка видео {video_idx + 1}/{self.meta_data['num_videos']}")
            if not self.meta_data['was_video_downloaded'][video_idx]:
                self.download_video(video_idx)
                # pylint: disable=logging-fstring-interpolation
                log.info(
                    f"Downloaded video: {sum(self.meta_data['was_video_downloaded']) + 1}/{self.meta_data['num_videos']}")  # pylint: disable=line-too-long
            if not self.meta_data['were_features_extracted'][video_idx]:
                self.extract_features_from_video(video_idx)
                # pylint: disable=logging-fstring-interpolation
                log.info(
                    f"Features extracted: {sum(self.meta_data['were_features_extracted']) + 1}/{self.meta_data['num_videos']}")  # pylint: disable=line-too-long
            if not self.meta_data['were_features_uploaded'][video_idx]:
                self.upload_features(video_idx)
                # pylint: disable=logging-fstring-interpolation
                log.info(
                    f"Uploaded features: {sum(self.meta_data['were_features_uploaded']) + 1}/{self.meta_data['num_videos']}")  # pylint: disable=line-too-long
        sort_dict_by_key(my_dict=self.meta_data, target_key='videos_duration')
        log.info("1 и 2 этапы пайплайна реализованы.")
        self.update_meta()

    def download_features_from_db(self, video_idx: int):
        """
        Функция загружает фич из БД по индексу в мета данных.
        Args:
            video_idx (int): Индекс видео из списка в мета данных.
        """
        self.minio_db.db_get_file(str(self.meta_data['remote_features_paths'][video_idx]),
                                  save_path=str(self.meta_data['local_features_paths'][video_idx]), bucket='tmp')

    def compare_video_and_main_video(self, video_idx: int, main_video_idx: int, group_idx_where_main: int) -> dict:
        """
        Функция сравнивает текущее видео (его фичи) с текущим главным видео (его фичами).
        Предполагается, что главное видео длиннее. После завершения работы функции мета данные обновляются.
        Args:
            video_idx (int): Индекс видео из списка в мета данных.
            main_video_idx (int): Индекс главного видео из списка в мета данных.
            group_idx_where_main (int): Индекс группы с которой соотносится главное видео.

        Returns:
            comparison_info (dict): Результат сравнения видео:
                    comparison_info['are_similar'] (bool): Похожи ли видео.
                    comparison_info['max_similarity'] (float): Максимальная достигнутая оценка схожести
                    comparison_info['was_main_compared_with_current_before'] (bool): Было ли текущее главное видео
                                                                                     сравнено с текущим видео ранее.
        """
        # скачивание
        if not self.meta_data['comparison_submeta'][video_idx]['was_main_video_downloaded'][group_idx_where_main]:
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info(f"\t\tDownloading main video...")
            self.download_features_from_db(main_video_idx)
            self.meta_data['comparison_submeta'][video_idx]['was_main_video_downloaded'][group_idx_where_main] = True
            self.update_meta()
        if not self.meta_data['comparison_submeta'][video_idx]['was_main_video_compared_with_current'][
            group_idx_where_main]:
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info(f"\t\tComparing main video...")
            short_video_info = {'features_path': self.meta_data['local_features_paths'][video_idx],
                                'duration': self.meta_data['videos_duration'][video_idx]}
            long_video_info = {'features_path': self.meta_data['local_features_paths'][main_video_idx],
                               'duration': self.meta_data['videos_duration'][main_video_idx]}
            comparison_result = self.model.compare_videos(short_video_info, long_video_info, self.model_threshold,
                                                          self.model_frames_step)
            comparison_result['was_main_compared_with_current_before'] = False
            os.remove(str(self.meta_data['local_features_paths'][main_video_idx]))
            self.meta_data['comparison_submeta'][video_idx]['was_main_video_compared_with_current'][
                group_idx_where_main] = True  # pylint: disable=line-too-long
            self.update_meta()
        else:
            comparison_result = {'was_main_compared_with_current_before': True}
        return comparison_result

    def compare_video_to_main_videos(self, video_idx):
        """
        Функция сравнивает текущее видео (его фичи) с текущими главными видео (их фичами).
        Предполагается, что все главные видео длиннее. После завершения работы функции мета данные обновляются.
        Кроме того, по завершении работы, функция проводит локальное удаление текущего видео, а также
        оставшихся главных.
        Args:
            video_idx: Индекс текущего видео из списка в мета данных.
        """

        if self.meta_data['was_video_with_error'][video_idx]:
            # video with error from the start
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info("\tCurrent video with error.")
            log.info("----------------------")
            self.meta_data['was_video_compared'] = True
            self.update_meta()
            return

        if self.meta_data['comparison_submeta'][video_idx] is None:
            # no comparison sybmeta data found
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info("\tInitializing comparison submeta for video...")
            self.meta_data['comparison_submeta'][video_idx] = init_submeta(
                main_videos_indices=self.meta_data['main_videos_in_groups_indices'],
                num_main_videos=self.meta_data['num_groups_found'],
                video_to_compare_idx=video_idx)
            self.update_meta()

        if not self.meta_data['comparison_submeta'][video_idx]['was_current_video_downloaded']:
            # not download of cur video found
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info("\tDownloading current video...")
            self.download_features_from_db(video_idx)
            self.meta_data['comparison_submeta'][video_idx]['was_current_video_downloaded'] = True
            self.update_meta()

        if not self.meta_data['comparison_submeta'][video_idx]['was_current_video_compared']:
            # cur video was not compared
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info("\tComparing current video and main videos...")
            for group_idx_where_main, main_video_idx in enumerate(
                    self.meta_data['comparison_submeta'][video_idx]['main_videos_indices']):
                # 2nd video is longer!
                # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
                log.info(
                    f"\tComparing main video {group_idx_where_main}/{self.meta_data['comparison_submeta'][video_idx]['num_main_videos']}")  # pylint: disable=line-too-long

                comparison_result = self.compare_video_and_main_video(video_idx, main_video_idx, group_idx_where_main)
                if not comparison_result['was_main_compared_with_current_before']:

                    if not comparison_result['are_similar']:
                        self.meta_data['comparison_submeta'][video_idx]["is_current_similar_to_main_videos"][
                            group_idx_where_main] = False
                    else:
                        # are similar
                        self.meta_data['groups_content_video_paths'][group_idx_where_main].append(
                            str(self.meta_data['remote_videos_paths'][video_idx]))  # pylint: disable=line-too-long
                        break
            if sum(self.meta_data['comparison_submeta'][video_idx]['is_current_similar_to_main_videos']) == 0:
                # if video is not in any group
                self.meta_data['main_videos_in_groups_indices'].append(video_idx)
                self.meta_data['main_videos_in_groups_videos_paths'].append(
                    self.meta_data['remote_videos_paths'][video_idx])  # pylint: disable=line-too-long
                self.meta_data['groups_content_video_paths'].append([])
                self.meta_data['num_groups_found'] += 1

            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info("\tUpdating meta for current video...")
            os.remove(str(self.meta_data['local_features_paths'][video_idx]))
            self.meta_data['comparison_submeta'][video_idx]['was_current_video_compared'] = True
            self.update_meta()

    def compare_videos(self):
        """
        Реализация 3 этапа пайплайна.
        Функция осуществляет сравнивание всех видео.
        После завершения работы функции мета данные обновляются.
        """
        for video_idx in range(self.meta_data['num_videos']):
            # pylint: disable=logging-fstring-interpolation, f-string-without-interpolation
            log.info(f"Comparing video {video_idx}/{self.meta_data['num_videos']}:")
            self.compare_video_to_main_videos(video_idx)
            log.info("Done.\n----------------------")
