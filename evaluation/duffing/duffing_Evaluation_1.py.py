"""
this file is for evaluating the results of Konzept1 System1
"""

import DKLc.networkloader as networkloader
import tensorflow as tf
import numpy as np
import pickle
import misc.helperfns as helperfns
import os
from pathlib import Path
# import matplotlib.pyplot as plt

filePath = Path(__file__)
experiments_full_Path = Path('experiments/duffing/model/duffing/uncorrected/addedStates_full/')

contents = os.listdir(filePath.parents[2] / experiments_full_Path)

pred_horizon = 50

for folder in contents:
    # TODO insert path with model file (same as inserted in params['model_path'])
    experimentPath = experiments_full_Path / folder
    # TODO insert path where to put the evaluation data
    saving_path = Path('duffing/p' + str(pred_horizon) + '_uncorrected_added_states_full/') / folder
    os.makedirs(saving_path, exist_ok=True)
    experimentPath = filePath.parents[2] / experimentPath
    params = pickle.load(open(experimentPath / Path('params.p'), 'rb'))

    sess = tf.Session()
    loader = networkloader.NetworkLoader(str(experimentPath) + '/', sess, params, load_tf_graph=False)

    # TODO insert length of trajectories - 1
    params['prediction_horizon'] = pred_horizon - 1
    if 'use_input_time_delay' not in params:
        params['use_input_time_delay'] = False
    if 'delay_coordinates_delay' in params:
        params['prediction_horizon'] -= params['delay_coordinates_delay']
        print(params['delay_coordinates_delay'])
        print('########################################################################')

    # TODO insert path to data. data should be in the form
    # <path_to_data>States.csv
    # <path_to_data>Inputs.csv
    path_to_data = str(filePath.parents[2] / Path('data/duffing/uncorrected/duffingOscillator_uncorrectedTest'))
    test_states, test_inputs = helperfns.prepare_data(path_to_data, params)
    width, height, depth = test_states.shape
    state_tensor = tf.placeholder(dtype=tf.float32, shape=(None, height, depth), name='states')
    width, height, depth = test_inputs.shape
    input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, height, depth), name='inputs')

    # predicted_states, lin_states, lin_states_ref = loader.predict_states(state_tensor, input_tensor)
    prediced_inputs = loader.input_decoder.get_net(loader.input_encoder.get_net(
        input_tensor[0:, :, 0:3]))
    lin_inputs = loader.input_encoder.get_net(
        input_tensor[0:, :, 0:3])

    if params['concept'] == 1:
        predicted_states, lin_states, lin_states_ref = loader.predict_states(state_tensor, input_tensor)
        test_states_pred = sess.run(predicted_states,
                                    feed_dict={state_tensor: test_states[0:, :, :],
                                               input_tensor: test_inputs[0:, :, :]})
    else:
        lin_states, lin_states_ref = loader.predict_states(state_tensor, input_tensor)


    test_lin_states = sess.run(lin_states,
                               feed_dict={state_tensor: test_states[0:, :, :], input_tensor: test_inputs[0:, :, :]})

    test_lin_states_ref = sess.run(lin_states_ref,
                                   feed_dict={state_tensor: test_states[0:, :, :], input_tensor: test_inputs[0:, :, :]})
    test_inputs_pred, test_lin_inputs = sess.run([prediced_inputs, lin_inputs],
                                                 feed_dict={input_tensor: test_inputs[0:, :, :]})
    n = 24
    if params['concept'] == 1:
        np.savetxt(saving_path / Path('TestTraj' + str(n) + '.csv'), test_states_pred[n, :, :])
    np.savetxt(saving_path / Path('TestTraj_ref' + str(n) + '.csv'), test_states[n, :, :])
    np.savetxt(saving_path / Path('TestTraj_lin' + str(n) + '.csv'), test_lin_states[n, :, :])
    np.savetxt(saving_path / Path('TestTraj_input' + str(n) + '.csv'), test_inputs[n, :, :])
    np.savetxt(saving_path / Path('TestTraj_lin_ref' + str(n) + '.csv'), test_lin_states_ref[n, :, :])
    #np.savetxt(saving_path / Path('TestTrajnonlinear.csv', test_states_nonlinear[n, :, :])
    np.savetxt(saving_path / Path('TestTraj_input_pred' + str(n) + '.csv'), test_inputs_pred[n, :, :])
    np.savetxt(saving_path / Path('TestTraj_input_lin' + str(n) + '.csv'), test_lin_inputs[n, :, :])

    if params['concept'] == 1:
        states_pred = test_states_pred[0, :, :]
    states = test_states[0, :, :]
    states_lin_ref = test_lin_states_ref[0, :, :]
    states_lin = test_lin_states[0, :, :]
    print(width)
    for i in range(1, width):
        if params['concept'] == 1:
            states_pred = np.concatenate((states_pred, test_states_pred[i, :, :]), axis=0)
        states = np.concatenate((states, test_states[i, :, :]), axis=0)
        states_lin_ref = np.concatenate((states_lin_ref, test_lin_states_ref[i, :, :]), axis=0)
        states_lin = np.concatenate((states_lin, test_lin_states[i, :, :]), axis=0)

    if params['concept'] == 1:
        np.savetxt(saving_path / Path('TestTrajTotal.csv'), states_pred)
    np.savetxt(saving_path / Path('TestTraj_refTotal.csv'), states)
    np.savetxt(saving_path / Path('TestTrajLinTotal.csv'), states_lin)
    np.savetxt(saving_path / Path('TestTrajLin_refTotal.csv'), states_lin_ref)

