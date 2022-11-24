"""
Модуль позволяющий сравнивать два видео, а также обрабатывать их.
"""
import numpy as np
import tensorflow as tf
from model.visil import ViSiL  # pylint: disable=import-error
from utils.manipulate_data import load_data  # pylint: disable=import-error


class VideoSimilarityModel:
    """
    Класс позволяющий обрабатывать и сравнивать видео.
    """

    def __init__(self, path_to_model: str):
        """
        Иннициализация класса для сравнения видео.
        Args:
            path_to_model: Путь до модели, осуществляющей сравнение видео.
        """
        tf.reset_default_graph()
        self.model = ViSiL(path_to_model)

    def extract_features(self, np_video: np.ndarray, batch_sz: int = 32):
        """
        Функция использующая модель ViSiL, чтобы вытянуть фичи из видео.
        Если batch_sz достаточно маленький, то это может вызвать проблемы с памятью.

        Args:
            np_video (np.ndarray): Видео представленное в numpy формате.
            batch_sz (int): Размер батча.

        Returns:
            Фичи, вытянутые из видео.
        """
        features = self.model.extract_features(np_video, batch_sz=batch_sz)
        return features

    def calculate_similarity(self, features_1: np.ndarray, features_2: np.ndarray) -> float:
        """
        Оценивание похожести частей видео по их фичам. Подразумевается, что размерности фич одинаковые.

        Args:
            features_1 (np.ndarray): Фичи части первого видео, представленные в numpy формате.
            features_2 (np.ndarray): Фичи части второго видео, представленные в numpy формате.

        Returns:
            weighted_average_sim_score (float): Оценка схожести фич.

        """

        weighted_average_sim_score = 0
        step = 500  # шаг с которым идет итерация по циклу, если будет слишком большим, то будет проблема с памятью
        len_features = len(features_1)

        for start in range(0, len_features, step):  # step 5000 is almost max valid
            features_1_crop = features_1[start: start + step, ...]
            features_2_crop = features_2[start: start + step, ...]
            len_crop = len(features_1_crop)

            similarity = self.model.calculate_video_similarity(features_1_crop, features_2_crop)
            weighted_average_sim_score += similarity * (len_crop / len_features)

        del features_1, features_2, features_1_crop, features_2_crop
        return weighted_average_sim_score

    def compare_videos(self, short_video_info: dict, long_video_info: dict, similarity_threshold: float,
                       step: int) -> dict:
        """
        Сравнение видео. Подразумевается, что long_video длиннее, чем short_video.
        Чем меньше step (шаг), тем точнее сравнение, но и тем медленнее будет выполняться функция.

        Сравнение делается следующим образом. Берем за нужную длину длину короткого видео. Затем делаем кроп
        из длинного видео длинной как у короткого видео начиная с индекса 0, сравниваем кроп и короткое видео,
        и если схожесть больше чем similarity_threshold, то считаем видео равными. Если же не схожи, то
        берем следующий кроп из длинного видео начиная с индекса 0 + step и т.д.

        Args:
            short_video_info (dict): Информация о коротком видео:
                                            short_video_info['features_path'] (str): Локальный путь до фич видео.
                                            short_video_info['duration'] (int): Длительность видео в секундах.
            long_video_info (dict): Информация о длинном видео:
                                            long_video_info['features_path'] (str): Локальный путь до фич видео.
                                            long_video_info['duration'] (int): Длительность видео в секундах.
            similarity_threshold (float): Пороговое значение для сравнения видео.
            step (int): Шаг с которым сдвигается индекс начала кропа из длинного видео (см. подробнее в описании).

        Returns:
            comparison_info (dict): Результат сравнения видео:
                                        comparison_info['are_similar'] (bool): Похожи ли видео.
                                        comparison_info['max_similarity'] (float): Максимальный достигнутая оценка
                                                                                    схожести

        """

        short_video_features = load_data(short_video_info['features_path'])
        long_video_features = load_data(long_video_info['features_path'])

        comparison_info = {'are_similar': False, 'max_similarity': 0}

        for i in range(0, long_video_info['duration'] - short_video_info['duration'], step):
            long_video_crop_features = long_video_features[i: i + short_video_info['duration'], ...]
            similarity = self.calculate_similarity(long_video_crop_features, short_video_features)
            del long_video_crop_features
            if similarity > comparison_info['max_similarity']:
                comparison_info['max_similarity'] = similarity
            if similarity >= similarity_threshold:
                comparison_info['are_similar'] = True
                break
        del short_video_features, long_video_features
        return comparison_info
