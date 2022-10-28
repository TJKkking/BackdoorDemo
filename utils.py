import numpy as np
from typing import Tuple

def preprocess(x, y, class_num: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: poisoned_data
    y: label
    class_num: 分类数0-9
    归一化
    """
    min_x, max_x = np.amin(x), np.amax(x)
    normal_x = (x - min_x) / (max_x - min_x)

    # categorical_y = to_categorical(y, class_num)
    labels = np.array(y, dtype = int)
    cate_y = np.zeros((labels.shape[0], class_num), dtype=np.float32)
    cate_y[np.arange(labels.shape[0]), np.squeeze(labels)] = 1

    return normal_x, cate_y