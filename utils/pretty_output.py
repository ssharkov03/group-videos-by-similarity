"""
Модуль для приведения выходных данных к читаемому виду.
"""
from utils.manipulate_data import load_data  # pylint: disable=import-error


def get_groups(meta_data: dict) -> dict:
    """
    Функция выделяет группы видео из мета данных.
    """
    ok_resp = dict()  # pylint: disable=use-dict-literal
    for idx, main_video in enumerate(meta_data['main_videos_in_groups_videos_paths']):
        ok_resp[main_video] = meta_data['groups_content_video_paths'][idx]
    resp = {
        'failed_to_process_videos': [meta_data['remote_videos_paths'][i] for i in range(meta_data['num_videos']) if
                                     meta_data['was_video_with_error']],
        'successful_comparison': ok_resp
    }
    return resp


def output_prettifier(path_to_output_meta: str,
                      save_path: str):
    """
    Функция приводит выходные данные к читаемому виду.
    Args:
        path_to_output_meta (str): Путь до выходных мета данных.
        save_path (str): Путь до txt файла, куда сохранять вывод.
    """
    with open(save_path, "w", encoding="utf8") as out:

        meta_data = load_data(path_to_output_meta)
        out.write(f"Сделано {len([i for i in meta_data['comparison_submeta'] if i is not None])} видео.\n")
        resp = get_groups(meta_data)
        good = resp['successful_comparison']
        for key, arr in good.items():
            if len(arr):
                out.write(f"Главное видео = {key}, его подмножества: \n")
                # pylint: disable=invalid-name
                for v in arr:
                    out.write(f"\t{v}\n")
                out.write("\n")
