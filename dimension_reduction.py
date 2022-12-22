import numpy as np
from data_loader import *


def reduce_dimension(raw: np.ndarray, reduction_policy='dim_nums', contribution_threshold=0.97, dim_nums=50):
    '''
    self-made PCA.
    reduction_policy can be set among ['dim_nums', 'contribution']
    if you choose 'dim_nums', then 'dim_nums' should be set,
    if you choose 'contribution', then 'contribution_threshold' should be set (0.0~1.0).
    '''
    B, N = raw.shape
    MEAN = raw.mean(axis=0)
    print(MEAN.shape)
    centered = raw - MEAN
    cov = np.dot(centered.T, centered)
    w, v = np.linalg.eig(cov)
    w = np.array([cplx.real for cplx in w])
    total = w.sum()
    contri = w / total
    accu_contribution = 0
    target_dim_nums = 0
    if reduction_policy == 'contribution':
        for i in range(len(contri)):
            accu_contribution += contri[i]
            target_dim_nums = i + 1
            if (accu_contribution >= contribution_threshold):
                break
    elif reduction_policy == 'dim_nums':
        target_dim_nums = min(N, dim_nums)
        for i in range(target_dim_nums):
            accu_contribution += contri[i]
    else:
        raise Exception('wrong policy')
    print(target_dim_nums)
    ev_map = v[:, :target_dim_nums]
    print(ev_map.shape)
    res = np.dot(centered, ev_map)
    return res, accu_contribution


def test():
    hog_land = load_data(arg_features='landmarks_and_hog')
    x = hog_land['X']
    y = hog_land['Y']
    print(x.shape)
    x, contri = reduce_dimension(x, reduction_policy='dim_nums', contribution_threshold=0.95, dim_nums=10)
    print(x.shape, contri)


if __name__ == '__main__':
    test()
