import time
import argparse
import pprint
import numpy as np 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from train import train
from parameters import HYPERPARAMS

# define the search space
fspace = {
    'theta':  hp.uniform('theta', 10, 30),
    'c_value': hp.uniform('c_value', 150, 250),
    'toler': hp.uniform('toler', 0.00008, 0.00012)
}

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--max_evals", required=True, help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
max_evals = int(args.max_evals)
current_eval = 1
train_history = []


def function_to_minimize(hyperparams, theta=HYPERPARAMS.theta, c_value=HYPERPARAMS.c_value, toler=HYPERPARAMS.toler):
    theta = hyperparams['theta']
    c_value = hyperparams['c_value']
    toler = hyperparams['toler']

    global current_eval 
    global max_evals
    print("#################################")
    print("       Evaluation {} of {}".format(current_eval, max_evals))
    print("#################################")
    start_time = time.time()
    try:
        accuracy = train(epochs=HYPERPARAMS.epochs_during_hyperopt, kernel=HYPERPARAMS.kernel,
                         theta=theta)
        training_time = int(round(time.time() - start_time))
        current_eval += 1
        train_history.append({'accuracy': accuracy, 'theta': theta,
                              'c_value': c_value, 'toler': toler, 'time': training_time})
    except Exception as e:
        print("#################################")
        print("Exception during training: {}".format(str(e)))
        print("Saving train history in train_history.npy")
        np.save("train_history.npy", train_history)
        exit()

    return {'loss': -accuracy, 'time': training_time, 'status': STATUS_OK}

# lunch the hyperparameters search


trials = Trials()
best_trial = fmin(fn=function_to_minimize, space=fspace, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# get some additional information and print( the best parameters
for trial in trials.trials:
    if trial['misc']['vals']['theta'][0] == best_trial['theta'] and \
            trial['misc']['vals']['c_value'][0] == best_trial['c_value'] and \
            trial['misc']['vals']['toler'][0] == best_trial['toler']:
        best_trial['accuracy'] = -trial['result']['loss'] * 100
        best_trial['time'] = trial['result']['time']

print("#################################")
print("      Best parameters found")
print("#################################")
pprint.pprint(best_trial)
print("#################################")
