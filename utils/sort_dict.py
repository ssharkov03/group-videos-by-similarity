"""
Утилита для сортировки словаря по ключу.
"""


def sort_dict_by_key(my_dict, target_key):
    """
    Функция для сортировки словаря по ключу.
    Если my_dict изменяется, то и исходный тоже изменится.
    """

    target_like_keys = [target_key]
    print(target_like_keys)
    for key in my_dict.keys():
        if isinstance(my_dict[key], list) and len(my_dict[key]) == len(my_dict[target_key]) and key != target_key:
            target_like_keys.append(key)

    sorted_content = [[] for i in range(len(target_like_keys))]

    for item in sorted(zip(*[my_dict[key] for key in target_like_keys]), reverse=True):
        for i, _ in enumerate(item):
            sorted_content[i].append(item[i])  # pylint: disable=unnecessary-list-index-lookup

    for key_idx, _ in enumerate(target_like_keys):
        my_dict[target_like_keys[key_idx]] = sorted_content[key_idx]
