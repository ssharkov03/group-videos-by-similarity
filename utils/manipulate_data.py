"""
Модуль, выполняющий локальное считывание и сохранение данных.
"""
import pickle
import cv2
import numpy as np


def save_data(data, save_path):
    """Сохранение объекта в pickle"""
    with open(save_path, 'wb') as output:
        pickle.dump(data, output)


def load_data(load_path):
    """Считывание объекта pickle"""
    with open(load_path, 'rb') as data:
        loaded_data = pickle.load(data)
        return loaded_data


def resize_frame(frame, desired_size):
    """Resizing кадра"""
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    # pylint: disable=no-member
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def center_crop(frame, desired_size):
    """Центральный кроп кадра"""
    old_size = frame.shape[:2]
    top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
    left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
    return frame[top: top + desired_size, left: left + desired_size, :]


def load_video(video, all_frames=False):
    """Функция для считывания локального видео в np.ndarray"""
    cv2.setNumThreads(3)  # pylint: disable=no-member
    cap = cv2.VideoCapture(video)  # pylint: disable=no-member
    fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member
    if fps > 144 or fps is None:
        fps = 25
    frames = []
    count = 0
    while cap.isOpened():
        if int(count % round(fps)) == 0 or all_frames:
            _, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
                frames.append(center_crop(resize_frame(frame, 256), 256))
            else:
                break
        count += 1
    cap.release()
    return np.array(frames)
