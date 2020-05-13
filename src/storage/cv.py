from __future__ import annotations
import numpy as np
import cv2
from src.storage.datasetrepo import GTSRBRepository


def pre_process_image(image: np.ndarray) -> np.ndarray:
    """
    1. histogram equalization
    2. normalization
    """
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = image/255.-.5
    return image


def transform_image(image, ang_range, shear_range, trans_range):
    """
    1. Rotation
    2. Transformation
    3. Shearing
    """
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows, cols, ch = image.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, Rot_M, (cols, rows))
    image = cv2.warpAffine(image, Trans_M, (cols, rows))
    image = cv2.warpAffine(image, shear_M, (cols, rows))
    image = pre_process_image(image)
    return image


def get_index_dict(y_train):
    """
    Returns indices of each label
    Assumes that the labels are 0 to N-1
    """
    dict_indices = {}
    ind_all = np.arange(len(y_train))
    for i in range(len(np.unique(y_train))):
        ind_i = ind_all[y_train == i]
        dict_indices[i] = ind_i
    return dict_indices


def gen_transformed_data(X_train, y_train, n_each, ang_range, shear_range, trans_range, randomize_Var):
    dict_indices = get_index_dict(y_train)
    n_class = len(np.unique(y_train))
    X_arr = []
    Y_arr = []
    for i in range(n_class):
        len_i = len(dict_indices[i])
        ind_rand = np.random.randint(0, len_i, n_each)
        ind_dict_class = dict_indices[i]

        for i_n in range(n_each):
            img_trf = transform_image(X_train[ind_dict_class[ind_rand[i_n]]], ang_range, shear_range, trans_range)
            X_arr.append(img_trf)
            Y_arr.append(i)

    X_arr = np.array(X_arr, dtype=np.float32())
    Y_arr = np.array(Y_arr, dtype=np.float32())

    if randomize_Var == 1:
        len_arr = np.arange(len(Y_arr))
        np.random.shuffle(len_arr)
        X_arr[len_arr] = X_arr
        Y_arr[len_arr] = Y_arr
    return X_arr, Y_arr


def pre_precess_gtsrb(gen_size: int):
    x_train, y_train, x_test, y_test = GTSRBRepository.load_from_pickle(is_jitter=False)
    jx_train, jy_train = gen_transformed_data(x_train, y_train, gen_size, 40, 5, 5, 1)
    rx_test = np.array([pre_process_image(x_test[i]) for i in range(len(x_test))])
    GTSRBRepository.save_as_pickle(filename='jitter_train', data={
        'features': jx_train, 'labels': jy_train
    })
    GTSRBRepository.save_as_pickle(filename='jitter_test', data={
        'features': rx_test, 'labels': y_test
    })
