import time
import argparse

from libSVM import LibSVM
from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS


def train(epochs=HYPERPARAMS.epochs, kernel=HYPERPARAMS.kernel, theta=HYPERPARAMS.theta,
          c_value=HYPERPARAMS.c_value, toler=HYPERPARAMS.toler, train_model=True):

        print( "loading dataset " + DATASET.name + "...")
        if train_model:
                data, validation = load_data(validation=True)
        else:
                data, validation, test = load_data(validation=True, test=True)
        
        if train_model:
            # Training phase
            print( "building model...")
            model_new = LibSVM(list(data['X']), list(data['Y']), c_value, toler, epochs, name=kernel, theta=theta)

            print("start training...")
            print("--")
            print("kernel: {}".format(kernel))
            print("theta: {} ".format(theta))
            print("max epochs: {} ".format(epochs))
            print("c_value: {} ".format(c_value))
            print("toler: {} ".format(toler))
            print("--")
            print("Training samples: {}".format(len(data['Y'])))
            print("Validation samples: {}".format(len(validation['Y'])))
            print("--")

            start_time = time.time()
            model_new.train()
            training_time = time.time() - start_time
            print("training time = {0:.1f} sec".format(training_time))

            if TRAINING.save_model:
                print("saving model...")
                model_new.save(TRAINING.save_model_path)

            print("evaluating...")
            validation_accuracy = model_new.predict(list(validation['X']), list(validation['Y']))
            print("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))
            return validation_accuracy
        else:
            # Testing phase : load saved model and evaluate on test dataset
            print( "start evaluation...")
            print( "loading pretrained model...")
            model_new = LibSVM.load(TRAINING.save_model_path)

            print("--")
            print("Validation samples: {}".format(len(validation['Y'])))
            print("Test samples: {}".format(len(test['Y'])))
            print("--")
            print("evaluating...")
            start_time = time.time()
            validation_accuracy = model_new.predict(list(validation['X']), list(validation['Y']))
            print("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))

            test_accuracy = model_new.predict(list(test['X']), list(test['Y']))
            print("  - test accuracy = {0:.1f}".format(test_accuracy*100))
            print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
            return test_accuracy


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
