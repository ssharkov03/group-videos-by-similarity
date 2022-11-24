"""
Модуль для приведения выходных данных к читаемому виду.
"""
import os

from utils.manipulate_data import load_data, save_data  # pylint: disable=import-error


def fix_download_bug():
    """
    Функция правит мета данные при некорректном скачивании данных с БД.
    """

    meta_data = load_data('output/meta_data_latest.pkl')
    saved_videos = list(os.listdir("saved_data"))
    saved_videos.remove('.ipynb_checkpoints')
    latest_idx = len([i for i in meta_data['comparison_submeta'] if i is not None]) - 1
    video_not_to_delete = os.path.split(meta_data['local_features_paths'][latest_idx])[-1]
    saved_videos.remove(video_not_to_delete)
    os.remove(os.path.join("saved_data", saved_videos[0]))
    # pylint: disable=invalid-name
    for idx, (z1, z2) in enumerate(zip(meta_data['comparison_submeta'][latest_idx]['was_main_video_downloaded'],
                                       meta_data['comparison_submeta'][latest_idx][
                                           'was_main_video_compared_with_current'])):
        if z1 != z2:
            meta_data['comparison_submeta'][latest_idx]['was_main_video_downloaded'][idx] = False
            save_data(meta_data, 'output/meta_data_latest.pkl')
            break
