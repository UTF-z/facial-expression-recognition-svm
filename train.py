import time
import argparse
import numpy as np

from libSVM import LibSVM
from data_loader import load_data
from parameters import DATASET, TRAINING, HYPERPARAMS
from dimension_reduction import reduce_dimension


def exp_train(hyper, training):
    epochs = hyper.epochs
    kernel = hyper.kernel
    theta = hyper.theta
    c_value = hyper.c_value
    toler = hyper.toler
    dim_reduction = hyper.dim_reduction
    data, validation = load_data(validation=True)
    if dim_reduction:
        data['X'], _ = reduce_dimension(data['X'],
                                        hyper.reduction_policy,
                                        hyper.contribution_threshold,
                                        hyper.dim_nums,
                                        val_data=validation)
    feature_dim = data['X'].shape[1]
    model_new = LibSVM(data['X'], data['Y'], c_value, toler, epochs, name=kernel, theta=theta)
    start_time = time.time()
    model_new.train()
    training_time = time.time() - start_time

    if training.save_model:
        model_new.save(training.save_model_path)

    training_accuracy = model_new.predict(data['X'], data['Y'])
    start_time = time.time()
    validation_accuracy = model_new.predict(validation['X'], validation['Y'])
    predict_time = time.time() - start_time
    exp_result = {
        'dim_reduction': dim_reduction,
        'feature_dim': feature_dim,
        'training_time': training_time,
        'validation_accuracy': validation_accuracy,
        'training_accuracy': training_accuracy,
        'predicting_time': predict_time
    }
    return exp_result


def train(epochs=HYPERPARAMS.epochs,
          kernel=HYPERPARAMS.kernel,
          theta=HYPERPARAMS.theta,
          c_value=HYPERPARAMS.c_value,
          toler=HYPERPARAMS.toler,
          dim_reduction=HYPERPARAMS.dim_reduction,
          train_model=True):
    print("loading dataset " + DATASET.name + "...")
    if train_model:
        data, validation = load_data(validation=True)
        if dim_reduction:
            data['X'], _ = reduce_dimension(data['X'], HYPERPARAMS.reduction_policy, HYPERPARAMS.contribution_threshold,
                                            HYPERPARAMS.dim_nums)
            validation['X'], _ = reduce_dimension(validation['X'], HYPERPARAMS.reduction_policy,
                                                  HYPERPARAMS.contribution_threshold, HYPERPARAMS.dim_nums)
    else:
        data, validation, test = load_data(validation=True, test=True)
        if dim_reduction:
            data['X'], _ = reduce_dimension(data['X'], HYPERPARAMS.reduction_policy, HYPERPARAMS.contribution_threshold,
                                            HYPERPARAMS.dim_nums)
            validation['X'], _ = reduce_dimension(validation['X'], HYPERPARAMS.reduction_policy,
                                                  HYPERPARAMS.contribution_threshold, HYPERPARAMS.dim_nums)
            test['X'], _ = reduce_dimension(test['X'], HYPERPARAMS.reduction_policy, HYPERPARAMS.contribution_threshold,
                                            HYPERPARAMS.dim_nums)

    print(f"feature_dim = {data['X'].shape[1]}")
    if train_model:
        # Training phase
        print("building model...")
        model_new = LibSVM(data['X'], data['Y'], c_value, toler, epochs, name=kernel, theta=theta)

        print("start training...")
        print("--")
        print(f"kernel: {kernel}")
        print(f"theta: {theta} ")
        print(f"max epochs: {epochs} ")
        print(f"c_value: {c_value} ")
        print(f"toler: {toler} ")
        print("--")
        print(f"Training samples: {len(data['Y'])}")
        print(f"Validation samples: {len(validation['Y'])}")
        print("--")

        start_time = time.time()
        model_new.train()
        training_time = time.time() - start_time
        print("training time = {0:.1f} sec".format(training_time))

        if TRAINING.save_model:
            print("saving model...")
            model_new.save(TRAINING.save_model_path)

        print("evaluating...")
        validation_accuracy = model_new.predict(validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        return validation_accuracy
    else:
        # Testing phase : load saved model and evaluate on test dataset
        print("start evaluation...")
        print("loading pretrained model...")
        model_new = LibSVM.load(TRAINING.save_model_path)

        print("--")
        print("Validation samples: {}".format(len(validation['Y'])))
        print("Test samples: {}".format(len(test['Y'])))
        print("--")
        print("evaluating...")
        start_time = time.time()
        validation_accuracy = model_new.predict(validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))

        test_accuracy = model_new.predict(test['X'], test['Y'])
        print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
        print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
        return test_accuracy


if __name__ == '__main__':
    # parse arg to see if we need to launch training now or not yet
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", default="yes", help="if 'yes', launch training from command line")
    parser.add_argument("-e", "--evaluate", default="yes", help="if 'yes', launch evaluation on test dataset")
    parser.add_argument("-m", "--max_evals", default="1")
    args = parser.parse_args()
    if args.train == "yes" or args.train == "Yes" or args.train == "YES":
        train()
    if args.evaluate == "yes" or args.evaluate == "Yes" or args.evaluate == "YES":
        train(train_model=False)
