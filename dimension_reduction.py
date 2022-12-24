import numpy as np
from data_loader import load_data
from sklearn.decomposition import PCA


def reduce_dimension(train_data: np.ndarray,
                     reduction_policy='dim_nums',
                     contribution_threshold=0.97,
                     dim_nums=50,
                     val_data=None):
    '''
    self-made PCA.
    reduction_policy can be set among ['dim_nums', 'contribution']
    if you choose 'dim_nums', then 'dim_nums' should be set,
    if you choose 'contribution', then 'contribution_threshold' should be set (0.0~1.0).
    '''
    B, N = train_data.shape
    MEAN = train_data.mean(axis=0)
    # STD = np.std(train_data, axis=0)
    centered = train_data - MEAN
    cov = np.dot(centered.T, centered)
    w, v = np.linalg.eig(cov)
    w = w.real
    v = v.real
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
        if target_dim_nums < 0:
            target_dim_nums = N
        for i in range(target_dim_nums):
            accu_contribution += contri[i]
    else:
        raise Exception('wrong policy')
    print(f"target_dim_nums = {target_dim_nums}")
    ev_map = v[:, :target_dim_nums]
    res = np.dot(centered, ev_map)
    if val_data is not None:
        val_centered = (val_data['X'] - MEAN)
        val_data['X'] = np.dot(val_centered, ev_map)
    return res, accu_contribution


def test():
    hog_land = load_data(arg_features='landmarks_and_hog')
    x = hog_land['X'][:10]
    y = hog_land['Y'][:10]
    self_made, contri = reduce_dimension(x, reduction_policy='contribution', contribution_threshold=0.95, dim_nums=10)
    pca = PCA(self_made.shape[1])
    std = pca.fit_transform(x)
    print(np.abs(np.abs(std) - np.abs(self_made)) < 1e-6)


if __name__ == '__main__':
    test()
