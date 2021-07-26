"""
This file gives a minimal example to estimate Koopman linaer systems using DKLc

We give four examples with and without state feedtrough and input encoding

Example 1:
- no feedthrough, no input encoding

Example 2:
- no feedthrough, input encoding

Example 3:
- state feedthrough, no input encoding

Example 4:
- state feedthrough, input encoding
"""

import DKLc.training as train
import random
import numpy as np
import tensorflow as tf
import os
import pickle
from shutil import copyfile

# set constant params
params = {}
params['training_data_file'] = 'training.csv'  # deprecated
params['delay_coordinates_delay'] = 0  # delay of time delay coordinates (0 for no time delay coordinates)

# TODO insert length of trajectory - 1
params['prediction_horizon'] = 49  # length of trajectories - 1
params['prediction_horizon'] = params['prediction_horizon'] - params['delay_coordinates_delay']

# True if information about e.g. the size of weights etc. should be logged in console
params['output_information'] = True
params['choose_state_prediction_loss'] = True  # use state prediction loss term
params['choose_Linf_state_prediction_loss'] = True  # use state prediction loss term with inf norm
params['choose_state_encoder_loss'] = True  # use state encoder loss term
params['choose_input_prediction_loss'] = True  # use input prediction loss term
params['choose_lin_state_prediction_loss'] = True  # use linear state prediction loss term
params['choose_Linf_lin_state_prediction_loss'] = False
params['choose_lin_input_prediction_loss'] = False  # use linear input prediction loss term
params['choose_dmd_loss'] = True  # use dmd loss
params['choose_l1_reg'] = True  # use l1 weight regularization
params['choose_l2_reg'] = True  # use l2 weight regularization
params['choose_decaying_weights'] = False  # use decaying weights like in Otto et al.
params['choose_dropout'] = False
dropout_rate = params['dropout_rate'] = 0.1
params['choose_sin_decoder'] = False  # choose sin layer in decoder (deprecated)
params['choose_constant_system_matrix'] = False  # choose constant matrix (deprecated)
params['regularize_loss'] = False  # choose to regularize prediction loss
params['regularize_loss_lin'] = False  # regularize linear prediction loss
params['train_input_autoencoder_only'] = False
params['predict_input'] = False

# training timing data
params['max_num_epochs'] = 50000
params['patience'] = 10000
params['opt_alg'] = 'adam'
params['learning_rate'] = 1e-4
params['use_input_time_delay'] = False
params[
    'use_cyclic_learning_rate'] = False  # if true learning rate will oscilate between params['learning_rate_max'] and
# params['learning_rate'] with frequency 2 * params['cylic_leaning_rate_length']
params['learning_rate_max'] = 1e-3
params['cylic_leaning_rate_length'] = 6
params['initialize_zero'] = False
params['batch_size'] = 512

params['num_epochs_train_encoder'] = 0  # (deprecated)

# TODO insert path, where the model should be saved in
params['model_path'] = 'vanDerPol/'
model_path = params['model_path']

# TODO insert path to training data Training data should be in the form
# <path_to_data>TrainingStates.csv
# <path_to_data>TrainingInputs.csv
# <path_to_data>ValidationStates.csv
# <path_to_data>ValidationInputs.csv
path_to_data = 'data/vanDerPol/vanDerPol_Oszillator'

params['log_loss'] = True  # log state predition loss etc.

# set system params
params['use_input_time_delay'] = False  # has to be set true if time delay coordinates should be used

# TODO insert the number of states that are measured
#  (if you want to measure every state simply count from 0 to length(states) - 1)
params['system_outputs'] = [0, 1]  # set outputs that are measured

# TODO insert number of inpus
params['dim_system_input'] = 1

if params['use_input_time_delay']:
    params['dim_system_input'] = params['dim_system_input'] * (params['delay_coordinates_delay'] + 1)
w_input = params['dim_system_input']
params['dim_system_state'] = len(params['system_outputs']) * (params['delay_coordinates_delay'] + 1)
w_state = params['dim_system_state']
params['is_input_affine'] = True  # this will toggle wether the input encoder is dependent on x (if false) or not

# loop over random params and train the nets
number_samples = 100

do_random_search = False
max_attempts_per_net = 2  # train the same net multiple times to avoid dependence on starting values for random search

# init random generator
random.seed()

best_loss = 0
best_net_number = 0
best_net_attempt = 0

# TODO insert dimensions of internal linear system
params['dim_lin_system'] = 9
w_lin_state = params['dim_lin_system']

# TODO select concept (1: full nonlinear transformation, 2: state feedthrough)
# Example 1 and 2
params['concept'] = 1

# Example 3 and 4
# params['concept'] = 2

# TODO set net structure !! do not remove w_* they are necessary
# State encoding
# Example 1 and 2
params['widths_state_encoder'] = [w_state, 20, 40, 60, w_lin_state]
params['widths_state_decoder'] = [w_lin_state, 40, 60, 80, len(params['system_outputs'])]

# Example 3 and 4
# params['widths_state_encoder'] = [w_state, 20, 40, 60, w_lin_state - params['system_outputs']]
# params['widths_state_decoder'] = [2]

# Input encoding
# Example 1 and 3
params['widths_input_encoder'] = [params['dim_system_input']]
params['widths_input_decoder'] = [params['dim_system_input']]

# Example 2 and 4
# params['widths_input_encoder'] = [w_input, 10, 20, w_input]
# params['widths_input_decoder'] = [w_input, 10, 20, w_input]

# TODO define if initial values shall be saved or loaded
params['save_initial_weights'] = False
params['initial_weights'] = None
# if initial value shall be used enter a path otherwise None
if not params['save_initial_weights']:
    params['initial_weights'] = 'experiments/vanDerPol/initial_weights/'

# TODO set loss hyperparameters (these standard parameters may lead to a good result)
params['alpha_state_prediction_loss'] = 1.0
params['delta_state_prediction_loss'] = 0.99  # decaying weight (like in Otto et al.) (note used when
# params['choose_decaying_weights'] is false)
params['alpha_state_encoder_loss'] = 1.0
params['alpha_input_prediction_loss'] = 1.0
params['delta_input_prediction_loss'] = 0.99
params['alpha_lin_state_prediction_loss'] = 1.0
params['delta_lin_state_prediction_loss'] = 0.99
params['alpha_lin_input_prediction_loss'] = 0.0  # (deprecated)
params['delta_lin_input_prediction_loss'] = 0.99  # (deprecated)
params['alpha_dmd_loss'] = 0.1
params['alphal1'] = 1e-6
params['alphal2'] = 1e-6
params['alpha_input_distance_loss'] = 1.0
params['alpha_state_distance_loss'] = 0.001
params['choose_dist_loss'] = False  # (deprecated)
params['Q_matrix'] = np.array([[40, 0, 0],
                               [0, 40, 0],
                               [0, 0, 4]])
params['R_matrix'] = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
params['Linf_state_prediction_loss_loss'] = 0.01

os.makedirs(params['model_path'], exist_ok=True)
trainer = train.Trainer(params, path_to_data, name='')
trainer.train()
