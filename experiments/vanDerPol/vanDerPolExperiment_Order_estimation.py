"""
This file contains the hyperparameter search and weight training for konzept 1
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
params['concept'] = 1
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
params['choose_decaying_weights'] = True  # use decaying weights like in Otto et al.
params['choose_dropout'] = False
dropout_rate = params['dropout_rate'] = 0.1
params['choose_sin_decoder'] = False  # choose sin layer in decoder (deprecated)
params['choose_constant_system_matrix'] = False  # choose constant matrix (deprecated)
params['regularize_loss'] = False  # choose to regularize prediction loss
params['regularize_loss_lin'] = False  # regularize linear prediction loss
params['train_input_autoencoder_only'] = False
params['predict_input'] = False

# training timing data
params['max_num_epochs'] = 1
params['patience'] = 10000
params['opt_alg'] = 'adam'
params['learning_rate'] = 1e-4
params['use_input_time_delay'] = False
params['use_cyclic_learning_rate'] = False  # if true learning rate will oscilate between
# params['learning_rate_max'] and params['learning_rate'] with frequency 2 * params['cylic_leaning_rate_length']
params['learning_rate_max'] = 1e-3
params['cylic_leaning_rate_length'] = 6
params['initialize_zero'] = False
params['batch_size'] = 512

params['num_epochs_train_encoder'] = 0  # (deprecated)

# TODO insert path, where the model should be saved in
params['model_path'] = 'models/'
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
params['dim_lin_system'] = 3
w_lin_state = params['dim_lin_system']

# TODO set net structure !! do not remove w_* they are necessary
params['widths_state_encoder'] = [w_state, 40, 60, 80, w_lin_state]
params['widths_input_encoder'] = [params['dim_system_input']]  # [w_input, 10, 20, w_input]
params['widths_state_decoder'] = [w_lin_state, 40, 60, 80, len(params['system_outputs'])]
params['widths_input_decoder'] = [params['dim_system_input']]  # [w_input, 10, 20, w_input]

# TODO define if initial values shall be saved or loaded
params['save_initial_weights'] = True
params['initial_weights'] = None

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
params['Q_matrix'] = np.array([[40, 0],
                               [0, 40]])
params['R_matrix'] = np.array([1])
params['Linf_state_prediction_loss_loss'] = 0.01

# [path, concept, koopman order, state encoder strcuture, state decoder strcuture,
# input encoder strcuture, input decoder strcuture]
looper_params = [['models/S3_n_2/', 1, 2, [w_state, 20, 40, 60, 2],
                  [2, 20, 40, 60, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_2/', 1, 2, [w_state, 20, 40, 60, 2],
                  [2, 20, 40, 60, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_3/', 1, 3, [w_state, 20, 40, 60, 3],
                  [3, 20, 40, 60, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_3/', 1, 3, [w_state, 20, 40, 60, 3],
                  [3, 20, 40, 60, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_3/', 2, 3, [w_state, 20, 40, 60, 3 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_3/', 2, 3, [w_state, 20, 40, 60, 3 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_4/', 1, 4, [w_state, 20, 40, 60, 4],
                  [4, 20, 40, 60, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_4/', 1, 4, [w_state, 20, 40, 60, 4],
                  [4, 20, 40, 60, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_4/', 2, 4, [w_state, 20, 40, 60, 4 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_4/', 2, 4, [w_state, 20, 40, 60, 4 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_5/', 1, 5, [w_state, 20, 40, 60, 5],
                  [5, 20, 40, 60, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_5/', 1, 5, [w_state, 20, 40, 60, 5],
                  [5, 20, 40, 60, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_5/', 2, 5, [w_state, 20, 40, 60, 5 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_5/', 2, 5, [w_state, 20, 40, 60, 5 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_6/', 1, 6, [w_state, 20, 40, 80, 6],
                  [6, 20, 40, 80, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_6/', 1, 6, [w_state, 20, 40, 80, 6],
                  [6, 20, 40, 80, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_6/', 2, 6, [w_state, 20, 40, 80, 6 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_6/', 2, 6, [w_state, 20, 40, 80, 6 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_7/', 1, 7, [w_state, 20, 40, 80, 7],
                  [7, 20, 40, 80, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_7/', 1, 7, [w_state, 20, 40, 80, 7],
                  [7, 20, 40, 80, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_7/', 2, 7, [w_state, 20, 40, 80, 7 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_7/', 2, 7, [w_state, 20, 40, 80, 7 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_8/', 1, 8, [w_state, 20, 40, 80, 8],
                  [8, 20, 40, 80, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_8/', 1, 8, [w_state, 20, 40, 80, 8],
                  [8, 20, 40, 80, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_8/', 2, 8, [w_state, 20, 40, 80, 8 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_8/', 2, 8, [w_state, 20, 40, 80, 8 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_9/', 1, 9, [w_state, 20, 40, 100, 9],
                  [9, 20, 40, 100, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_9/', 1, 9, [w_state, 20, 40, 100, 9],
                  [9, 20, 40, 100, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_9/', 2, 9, [w_state, 20, 40, 100, 9 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_9/', 2, 9, [w_state, 20, 40, 100, 9 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_10/', 1, 10, [w_state, 20, 40, 100, 10],
                  [10, 20, 40, 100, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_10/', 1, 10, [w_state, 20, 40, 100, 10],
                  [10, 20, 40, 100, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_10/', 2, 10, [w_state, 20, 40, 100, 10 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S2_n_10/', 2, 10, [w_state, 20, 40, 100, 10 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S3_n_20/', 1, 20, [w_state, 20, 60, 120, 20],
                  [20, 20, 60, 120, 2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/S4_n_20/', 1, 20, [w_state, 20, 60, 120, 20],
                  [20, 20, 60, 120, 2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]],
                 ['models/S1_n_20/', 2, 20, [w_state, 20, 60, 120, 20 - 2],
                  [2], [params['dim_system_input']], [params['dim_system_input']]],
                 ['models/vanDerPol/own_dataGet/S2_n_20/', 2, 20, [w_state, 20, 60, 120, 20 - 2],
                  [2], [w_input, 10, 20, w_input], [w_input, 10, 20, w_input]]
                 ]

for element in looper_params:
    params['model_path'] = element[0]
    params['concept'] = element[1]
    params['dim_lin_system'] = element[2]
    params['widths_state_encoder'] = element[3]
    params['widths_input_encoder'] = element[5]
    params['widths_state_decoder'] = element[4]
    params['widths_input_decoder'] = element[6]

    os.makedirs(params['model_path'], exist_ok=True)
    trainer = train.Trainer(params, path_to_data, name='')
    trainer.train()

    trainer = None
    tf.reset_default_graph()
