"""
this file contains the implementation of the neural network training including data preparation
"""

import tensorflow as tf
import numpy as np
import math
import DKLc.networkarch as net
import DKLc.lossarch as lossarch
import misc.helperfns as helperfns
import time
import DKLc.networkloader as networkloader
import os


class Trainer:

    def __init__(self, params, data_file_prefix, name, model_file=None, params_loader=None):
        """
        constructor of class Trainer. !!!! resets current tensorflow graph
        :param params: parameters needed
        :param data_file_prefix: prefix of files where training +validation data is. Data should be saved as
        training_data_file_prefix+<Training/Validation><States/Inputs/Time>.csv
        :param name: name of the trainer (will be added to all saved files)
        :param modelFile: file where a saved tensorflow model lies. If None Trainer will create a new model
        """
        # reset graph (just to avoid unexpected things to happen)
        tf.reset_default_graph()
        tf.keras.backend.clear_session()

        self.sess = tf.Session()

        self.params = params

        # create net architecture
        if model_file is None:
            self.nets = net.Nets(params)
        else:
            if params_loader is None:
                params_loader = params
            # loader = networkloader.NetworkLoader(model_file, self.sess, params_loader, load_tf_graph=False)
            if params['train_input_autoencoder_only']:
                self.nets = net.Nets(params, state_encoder_file=model_file + 'state_encoder',
                                     state_decoder_file=model_file + 'state_decoder',
                                     state_matrix_filename=model_file + 'K.csv')
            else:
                self.nets = net.Nets(params, state_encoder_file=model_file + 'state_encoder',
                                     input_encoder_file=model_file + 'input_encoder',
                                     state_decoder_file=model_file + 'state_decoder',
                                     input_decoder_file=model_file + 'input_decoder',
                                     state_matrix_filename=model_file + 'K.csv',
                                     input_vector_filename=model_file + 'L.csv')

        # create loss architecture
        self.loss = lossarch.Loss(params, self.nets)

        self.data_file_prefix = data_file_prefix
        self.name = name
        self.length_trajectories = self.params['prediction_horizon'] + 1

        # load training data
        self.training_states, self.training_inputs = helperfns.prepare_data(data_file_prefix + 'Training', self.params)
        self.num_training_trajectories, _, _ = self.training_states.shape  # number of trajectories
        self.num_training_batches = math.floor(self.num_training_trajectories / self.params['batch_size'])
        print('Loaded ' + str(self.num_training_trajectories) + ' trajectories for training')
        print('Data contains ' + str(self.num_training_batches) + ' batches for training')

        # load validation data
        self.validation_states, self.validation_inputs = helperfns.prepare_data(
            data_file_prefix + 'Validation', self.params)
        self.num_validation_trajectories, _, _ = self.validation_states.shape  # number of trajectories
        print('Loaded ' + str(self.num_validation_trajectories) + ' trajectories for validation')

        # counter and saver for training interruption based on validation loss
        self.num_epochs_without_validation_decrease = 0
        self.min_validation_loss = None

        self.num_epochs_after_save = 0

    def train(self):
        """
        trains nets
        :return:'state prediction loss
        """
        # get loss
        state_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.length_trajectories, self.params['dim_system_state']],
                                      name='states')
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.length_trajectories, self.params['dim_system_input']],
                                      name='inputs')
        learning_rate_tensor = tf.placeholder(dtype=tf.float32,
                                              shape=[],
                                              name='learning_rate')
        total_loss, state_prediction_loss, input_prediction_loss, lin_state_prediction_loss, \
        lin_input_prediction_loss, state_encoder_loss = self.loss.loss(state_tensor, input_tensor)

        trainable_var = tf.trainable_variables()
        if self.params['train_input_autoencoder_only']:
            trainable_var = self.nets.get_trainable_vars_input_encoder() + \
                            self.nets.get_trainable_vars_input_decoder()
            trainable_var.append(self.nets.get_lin_input_vector())

        # optimizer
        optimizer_total = helperfns.choose_optimizer(self.params, total_loss, trainable_var, name='Opt' + self.name,
                                                     learning_rate=learning_rate_tensor)
        optimizer_encoder = None
        if self.params['num_epochs_train_encoder'] > 0:
            optimizer_encoder = helperfns.choose_optimizer(self.params, state_encoder_loss, trainable_var,
                                                           name='Opt' + self.name,
                                                           learning_rate=learning_rate_tensor)

        # init tensorflow
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # save Initial Values
        if 'save_initial_weights' in self.params:
            if self.params['save_initial_weights']:
                os.makedirs(self.params['model_path'] + 'initial_weights/', exist_ok=True)
                self.nets.save_nets(self.params['model_path'] + 'initial_weights/', self.sess)

        # load Initial Values
        if 'initial_weights' in self.params:
            if self.params['initial_weights'] is not None:
                self.nets.load_nets(self.params['initial_weights'])
                init = tf.global_variables_initializer()
                self.sess.run(init)

        # init loss history
        validation_loss_history = []
        training_loss_history = []

        training_state_prediction_loss_history = []
        training_input_prediction_loss_history = []
        training_lin_state_prediction_loss_history = []
        training_lin_input_prediction_loss_history = []

        validation_state_prediction_loss_history = []
        validation_input_prediction_loss_history = []
        validation_lin_state_prediction_loss_history = []
        validation_lin_input_prediction_loss_history = []

        validation_loss = 0

        use_cyclic_learning_rate = False
        if 'use_cyclic_learning_rate' in self.params:
            use_cyclic_learning_rate = self.params['use_cyclic_learning_rate']
        use_decaying_learning_rate = False
        if 'use_decaying_learning_rate' in self.params:
            use_decaying_learning_rate = self.params['use_decaying_learning_rate']
        # loop over epochs
        for epoch in range(0, self.params['max_num_epochs']):
            millis = int(round(time.time() * 1000))

            # shuffle training data
            ind = np.arange(self.num_training_trajectories)
            np.random.shuffle(ind)
            epoch_training_states = self.training_states[ind, :, :]
            epoch_training_inputs = self.training_inputs[ind, :, :]

            # loop over steps
            # TODO some data are not hit in one epoch but due to random shuffle they will get hit
            training_loss = 0
            training_state_prediction_loss = 0
            training_input_prediction_loss = 0
            training_lin_state_prediction_loss = 0
            training_lin_input_prediction_loss = 0
            learning_rate = self.params['learning_rate']
            if use_decaying_learning_rate:
                learning_rate *= math.exp(-epoch * self.params['decaying_learning_rate_lambda'])
            if use_cyclic_learning_rate:
                learning_rate_max = self.params['learning_rate_max']
                if use_decaying_learning_rate:
                    learning_rate_max *= math.exp(-epoch * self.params['decaying_learning_rate_lambda'])
                cycle = math.floor(1 + epoch / (2 * self.params['cylic_leaning_rate_length']))
                x = abs(epoch / self.params['cylic_leaning_rate_length'] - 2 * cycle + 1)
                learning_rate = self.params['learning_rate'] + \
                                (learning_rate_max - self.params['learning_rate']) * max([0, 1 - x])
            loss = 0
            optimizer = 0
            # select other loss for training for the first steps (like in Lusch et al.)
            if epoch >= self.params['num_epochs_train_encoder']:
                optimizer = optimizer_total
                loss = total_loss
            else:
                optimizer = optimizer_encoder
                loss = state_encoder_loss
            for step in range(0, self.num_training_batches):
                # get batch out of data
                batch_training_states = epoch_training_states[
                                        step * self.params['batch_size']: (step + 1) * self.params['batch_size'], :, :]
                batch_training_inputs = epoch_training_inputs[
                                        step * self.params['batch_size']: (step + 1) * self.params['batch_size'], :, :]

                # run optimization
                step_loss = 0
                if not self.params['log_loss']:
                    _, step_loss = self.sess.run([optimizer, loss], feed_dict={state_tensor: batch_training_states,
                                                                               input_tensor: batch_training_inputs,
                                                                               learning_rate_tensor: learning_rate})
                else:
                    _, step_loss, step_state_prediction_loss, step_input_prediction_loss, \
                    step_lin_state_prediction_loss, step_lin_input_prediction_loss = self.sess.run(
                        [optimizer, loss, state_prediction_loss, input_prediction_loss, lin_state_prediction_loss,
                         lin_input_prediction_loss], feed_dict={state_tensor: batch_training_states,
                                                                input_tensor: batch_training_inputs,
                                                                learning_rate_tensor: learning_rate})
                    training_state_prediction_loss += step_state_prediction_loss
                    training_input_prediction_loss += step_input_prediction_loss
                    training_lin_state_prediction_loss += step_lin_state_prediction_loss
                    training_lin_input_prediction_loss += step_input_prediction_loss

                training_loss += step_loss

            # loss calculations
            training_loss /= self.num_training_batches
            if self.params['log_loss']:
                training_state_prediction_loss /= self.num_training_batches
                training_input_prediction_loss /= self.num_training_batches
                training_lin_state_prediction_loss /= self.num_training_batches
                training_lin_input_prediction_loss /= self.num_training_batches

                training_state_prediction_loss_history.append(training_state_prediction_loss)
                training_input_prediction_loss_history.append(training_input_prediction_loss)
                training_lin_state_prediction_loss_history.append(training_lin_state_prediction_loss)
                training_lin_input_prediction_loss_history.append(training_lin_input_prediction_loss)

            validation_loss = 0
            if not self.params['log_loss']:
                validation_loss = self.sess.run(loss, feed_dict={state_tensor: self.validation_states,
                                                                 input_tensor: self.validation_inputs})
            else:
                validation_loss, validation_state_prediction_loss, validation_input_prediction_loss, \
                validation_lin_state_prediction_loss, validation_lin_input_prediction_loss = self.sess.run(
                    [total_loss, state_prediction_loss, input_prediction_loss, lin_state_prediction_loss,
                     lin_input_prediction_loss], feed_dict={state_tensor: self.validation_states,
                                                            input_tensor: self.validation_inputs})

                print(validation_state_prediction_loss)
                print('IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
                validation_state_prediction_loss_history.append(validation_state_prediction_loss)
                validation_input_prediction_loss_history.append(validation_input_prediction_loss)
                validation_lin_state_prediction_loss_history.append(validation_lin_state_prediction_loss)
                validation_lin_input_prediction_loss_history.append(validation_lin_input_prediction_loss)

            # save loss in list
            validation_loss_history.append(validation_loss)
            training_loss_history.append(training_loss)

            # if validation loss decreased save model
            if self.min_validation_loss is not None:
                if 2 > self.min_validation_loss > validation_loss and self.num_epochs_after_save >= 0:
                    self.nets.save_nets(self.params['model_path'], self.sess)
                    helperfns.save_params(self.params, self.name)
                    self.save_loss(training_loss_history, validation_loss_history,
                                   validation_input_prediction_loss_history,
                                   validation_state_prediction_loss_history,
                                   validation_lin_state_prediction_loss_history)
                    self.num_epochs_after_save = 0
            self.num_epochs_after_save += 1
            if epoch > self.params['num_epochs_train_encoder']:
                # check if training should be stopped earlier cause of validation loss
                if not self.check_validation_loss_decrease(validation_loss):
                    print('Validation loss has stopped decreasing. aborting...')
                    validation_loss_history.append(validation_loss)
                    training_loss_history.append(training_loss)
                    break

            print('Training loss: ' + str(training_loss) + ', Validation loss: ' + str(validation_loss) +
                  ', Finished Epoch ' + str(epoch + 1) + '/' + str(self.params['max_num_epochs']) +
                  ', Needed ' + str(int(round(time.time() * 1000) - millis)) + 'ms for this epoch')

        # perform prediction on one validation trajectory (to get a quick trajectory to check net)
        lin_state_pred, state_pred, lin_system_input_pred, system_input_pred = \
            self.nets.get_prediction(tf.identity(state_tensor), tf.identity(input_tensor))

        out_state = self.sess.run(state_pred, feed_dict={state_tensor: self.validation_states[0:20, :, :],
                                                         input_tensor: self.validation_inputs[0:20, :, :]})
        np.savetxt(self.params['model_path'] + 'Verlauf' + self.name + '.csv', out_state[5, :, :])
        np.savetxt(self.params['model_path'] + 'Verlauf_ref' + self.name + '.csv', self.validation_states[5, :, :])

        # save important stuff
        # self.save_loss(training_loss_history, validation_loss_history,
        #               validation_input_prediction_loss_history,
        #               validation_state_prediction_loss_history,
        #               validation_lin_state_prediction_loss_history)
        # helperfns.save_params(self.params, self.name)

        # return prediction loss instead of validation loss
        return self.sess.run(state_prediction_loss, feed_dict={state_tensor: self.validation_states,
                                                               input_tensor: self.validation_inputs}), \
               self.sess.run(self.nets.get_lin_system_matrix())

    def check_validation_loss_decrease(self, validation_loss):
        """
        checks if validation loss decreased
        :param validation_loss:
        :return: true if validation loss has not decreased in the last params['patience'] epochs
        """
        if self.min_validation_loss is None:
            self.min_validation_loss = validation_loss
            return True
        elif self.min_validation_loss > validation_loss:
            self.num_epochs_without_validation_decrease = 0
            self.min_validation_loss = validation_loss
            return True
        self.num_epochs_without_validation_decrease += 1
        if self.num_epochs_without_validation_decrease >= self.params['patience']:
            return False
        return True

    def save_loss(self, train_loss, val_loss, input_loss, state_loss, lin_state_loss):
        np.savetxt(self.params['model_path'] + 'TrainingLoss' + self.name + '.csv', train_loss)
        np.savetxt(self.params['model_path'] + 'ValidationLoss' + self.name + '.csv', val_loss)
        np.savetxt(self.params['model_path'] + 'InputLoss' + self.name + '.csv', input_loss)
        np.savetxt(self.params['model_path'] + 'StateLoss' + self.name + '.csv', state_loss)
        np.savetxt(self.params['model_path'] + 'LinStateLoss' + self.name + '.csv', lin_state_loss)
