"""
Модуль, для создания структуры, позволяющей,
 хранить дополнительную информацию о состоянии текущего
 видео при сравнении с другими (главными).
"""
from typing import List


def init_submeta(main_videos_indices: List[int], video_to_compare_idx: int, num_main_videos: int) -> dict:
    """
    Инициализация структуры, позволяющей,
    хранить дополнительную информацию о состоянии текущего
    видео при сравнении с главными видео.

    Args:
        main_videos_indices (List[int]): Массив индексов из meta соответствующих главным видео.
        video_to_compare_idx (int): Индекс из meta текущего видео, которое будет сравниваться с главными.
        num_main_videos (int): Текущее количество главных видео.


    Returns:
        Словарь со следующими ключами:
            was_current_video_downloaded (bool): Было ли текущее видео скачано из БД (локальное наличие).
            was_current_video_compared (bool): Было ли текущее видео сравнено со всеми главными видео.
            current_video_idx (int): Индекс текущего видео в meta.
            main_videos_indices (List[int]): Массив индексов из meta соответствующих главным видео.
            was_main_video_downloaded (List[bool]): Было ли главное видео скачано из БД (локальное наличие).
            was_main_video_compared_with_current (List[bool]): Было ли текущее видео сравнено с главным.
            is_current_similar_to_main_videos (List[bool]): Результат сравнения текущего видео с главным.
            num_main_videos (int): Текущее количество главных видео.
    """

    return {'was_current_video_downloaded': False,
            'was_current_video_compared': False,
            'current_video_idx': video_to_compare_idx,
            'main_videos_indices': main_videos_indices,
            'was_main_video_downloaded': [False for _ in range(num_main_videos)],
            'was_main_video_compared_with_current': [False for _ in range(num_main_videos)],
            'is_current_similar_to_main_videos': [True for _ in range(num_main_videos)],
            'num_main_videos': num_main_videos}
