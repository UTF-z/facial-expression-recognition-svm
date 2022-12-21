class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013_features/Training'
    validation_folder = 'fer2013_features/PublicTest'
    test_folder = 'fer2013_features/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1


class Hyperparams:
    epochs = 10000
    epochs_during_hyperopt = 500
    kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
    features = 'landmarks_and_hog' # 'hog', 'landmarks', 'landmarks_and_hog'
    theta = 22.178  # default = 20
    c_value = 253.944   # default = 200
    toler = 0.00010735  # default = 0.0001


class Training:
    save_model = True
    save_model_path = "model.txt"


DATASET = Dataset()
TRAINING = Training()
HYPERPARAMS = Hyperparams()