'''
this file contains the implementation of the losses needed
'''

import tensorflow as tf
import numpy as np


class Loss:

    def __init__(self, params, nets):
        """
        constructor of class Loss
        :param params:
        :param nets: net architecture should be type networkarch.Nets
        """
        self.params = params
        self.nets = nets

        self.epsilon = 1e-5
        if 'epsilon' in params:
            self.epsilon = params['epsilon']

        self.use_lstm = False
        self.starting_point = 0
        if 'use_lstm_layer' in params:
            self.use_lstm = params['use_lstm_layer']
            if self.use_lstm:
                self.starting_point = self.params['rnn_start_length'] - 1

    def state_prediction_loss(self, state_pred, state):
        """
        calculates the prediciton loss for the states
        :param state_pred:
        :param state:
        :return: prediction loss for the states
        """
        loss = 0
        print(state.shape)
        print(state_pred.shape)
        raw_loss = tf.square(state_pred -
                             state[:, self.starting_point:,
                             0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))])
        if self.params['regularize_loss']:
            raw_loss = tf.divide(raw_loss, tf.square(
                state[:, :, 0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))]) +
                                 self.epsilon)
        if self.params['choose_decaying_weights']:
            # implements decaying weights like in Otto et al.
            decaying_weights = np.zeros((self.params['prediction_horizon'] + 1))
            sum_decaying_weights = 0
            for i in range(0, self.params['prediction_horizon'] + 1):
                decaying_weights[i] = self.params['delta_state_prediction_loss'] ** i
                sum_decaying_weights += self.params['delta_state_prediction_loss'] ** i

            # weighted median (multiply with n because tf.reduce_mean divides by n)
            decaying_weights = (self.params['prediction_horizon'] + 1) * \
                               (decaying_weights / sum_decaying_weights) * self.params['alpha_state_prediction_loss']
            decaying_weights_tensor = tf.constant(decaying_weights, dtype=tf.float32)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=0)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=2)
            loss = tf.reduce_mean(
                tf.multiply(decaying_weights_tensor, raw_loss), name='state_prediction_loss')
        else:
            alpha_tensor = tf.constant(self.params['alpha_state_prediction_loss'], dtype=tf.float32)
            loss = tf.multiply(alpha_tensor, tf.reduce_mean(raw_loss),
                               name='state_prediction_loss')

        if self.params['choose_Linf_state_prediction_loss']:
            # TODO inf norm of the vectors or 2 norm?
            loss_inf = 0
            loss_inf = tf.reduce_mean(tf.norm(tf.norm(
                state[:, self.starting_point:,
                      0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))] -
                state_pred,
                axis=2, ord=np.inf), axis=1, ord=np.inf))
            alpha_tensor = tf.constant(self.params['Linf_state_prediction_loss_loss'], dtype=tf.float32)
            return loss + tf.multiply(alpha_tensor, loss_inf)
        else:
            return loss

    def dmd_loss(self, lin_state, lin_system_input):
        lin_state_pred = self.nets.get_lin_model(lin_state[:, max(self.starting_point - 1, 0):, :], lin_system_input[:, self.starting_point:, :])
        _, width, _ = lin_state_pred.shape
        loss = tf.divide(
            tf.square(lin_state_pred[:, 0:width-1, :] - lin_state[:, max(self.starting_point, 1):, :]),
            tf.square(lin_state[:, max(self.starting_point, 1):, :]) + self.epsilon)
        alpha_tensor = tf.constant(self.params['alpha_dmd_loss'], dtype=tf.float32)
        return tf.multiply(alpha_tensor, tf.reduce_mean(loss),
                           name='dmd_loss')

    def state_encoder_loss(self, state_decoded, state):
        """
        calculates the loss for the encoder decoder
        :param state_decoded:
        :param state:
        :return:
        """
        raw_loss = tf.square(state_decoded -
                             state[:, self.starting_point:, 0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))])
        if self.params['regularize_loss']:
            raw_loss = tf.divide(raw_loss, tf.square(
                state[:, :, 0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))]) +
                                 self.epsilon)
        # does not make any sense doing a decaying weight because this one is not a loss for a data series
        alpha_tensor = tf.constant(self.params['alpha_state_encoder_loss'], dtype=tf.float32)
        return tf.multiply(alpha_tensor, tf.reduce_mean(raw_loss),
                           name='state_encoder_loss')

    def input_prediction_loss(self, system_input_pred, system_input):
        """
        calculates the prediction loss for the input
        :param system_input_pred:
        :param system_input:
        :return: prediction loss for the input
        """
        if self.params['choose_decaying_weights']:
            decaying_weights = np.zeros((self.params['prediction_horizon'] + 1))
            sum_decaying_weights = 0
            for i in range(0, self.params['prediction_horizon'] + 1):
                decaying_weights[i] = self.params['delta_input_prediction_loss'] ** i
                sum_decaying_weights += self.params['delta_input_prediction_loss'] ** i
            # s.t.
            # this operation is close to weighted median (multiply with n because reduce mean divides by n)
            decaying_weights = (self.params['prediction_horizon'] + 1) * \
                               (decaying_weights / sum_decaying_weights) * self.params['alpha_input_prediction_loss']
            decaying_weights_tensor = tf.constant(decaying_weights, dtype=tf.float32)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=0)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=2)
            return tf.reduce_mean(
                tf.multiply(decaying_weights_tensor, tf.square(system_input_pred - system_input)),
                name='input_prediction_loss')
        else:
            end = self.params['dim_system_input']
            if 'use_input_time_delay' in self.params:
                if self.params['use_input_time_delay']:
                    end = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
            alpha_tensor = tf.constant(self.params['alpha_input_prediction_loss'], dtype=tf.float32)
            return tf.multiply(alpha_tensor, tf.reduce_mean(tf.square(system_input_pred - system_input[:, :, 0:end])),
                               name='input_prediction_loss')

    def lin_state_prediction_loss(self, lin_state_pred, lin_state):
        """
        calculates the prediction loss for the linear states
        :param lin_state_pred:
        :param lin_state:
        :return: prediction loss for the linear states
        """
        loss = 0
        raw_loss = tf.square(lin_state_pred - lin_state[:, max(self.starting_point - 1, 0):, :])
        if self.params['regularize_loss_lin']:
            raw_loss = tf.divide(raw_loss, tf.square(lin_state[:, max(self.starting_point - 1, 0):, :]) + self.epsilon)
        if self.params['choose_decaying_weights']:
            decaying_weights = np.zeros((self.params['prediction_horizon'] + 1))
            sum_decaying_weights = 0
            for i in range(0, self.params['prediction_horizon'] + 1):
                decaying_weights[i] = self.params['delta_lin_state_prediction_loss'] ** i
                sum_decaying_weights += self.params['delta_lin_state_prediction_loss'] ** i
            # s.t.
            # this operation is close to weighted median (multiply with n because reduce mean divides by n)
            decaying_weights = (self.params['prediction_horizon'] + 1) * \
                               (decaying_weights / sum_decaying_weights) * self.params['alpha_lin_state_prediction_loss']
            decaying_weights_tensor = tf.constant(decaying_weights, dtype=tf.float32)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=0)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=2)
            loss = tf.reduce_mean(
                tf.multiply(decaying_weights_tensor, raw_loss),
                name='lin_state_prediction_loss')
        else:
            alpha_tensor = tf.constant(self.params['alpha_lin_state_prediction_loss'], dtype=tf.float32)
            loss = tf.multiply(alpha_tensor, tf.reduce_mean(raw_loss),
                               name='lin_state_prediction_loss')

        if self.params['choose_Linf_lin_state_prediction_loss']:
            loss_inf = tf.reduce_mean(
                tf.norm(tf.norm(lin_state[:, max(self.starting_point - 1, 0):, :]
                                - lin_state_pred, axis=2, ord=np.inf), axis=1, ord=np.inf))
            alpha_tensor = tf.constant(self.params['Linf_input_prediction_loss_loss'], dtype=tf.float32)
            return loss + tf.multiply(alpha_tensor, loss_inf)
        else:
            return loss

    def lin_input_prediction_loss(self, lin_system_input_pred, lin_system_input):
        """
        calculates the prediction loss for the linear states
        :param lin_system_input_pred:
        :param lin_system_input:
        :return: prediction loss for the linear states
        """

        if self.params['choose_decaying_weights']:
            decaying_weights = np.zeros((self.params['prediction_horizon'] + 1))
            sum_decaying_weights = 0
            for i in range(0, self.params['prediction_horizon'] + 1):
                decaying_weights[i] = self.params['delta_lin_input_prediction_loss'] ** i
                sum_decaying_weights += self.params['delta_lin_input_prediction_loss'] ** i
            # s.t.
            # this operation is close to weighted median
            decaying_weights = (self.params['prediction_horizon'] + 1) * \
                               (decaying_weights / sum_decaying_weights) * self.params['alpha_lin_input_prediction_loss']
            decaying_weights_tensor = tf.constant(decaying_weights, dtype=tf.float32)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=0)
            decaying_weights_tensor = tf.expand_dims(decaying_weights_tensor, axis=2)
            return tf.reduce_mean(
                tf.multiply(decaying_weights_tensor,
                            tf.square(lin_system_input_pred - lin_system_input)),
                name='lin_input_prediction_loss')
        else:
            alpha_tensor = tf.constant(self.params['alpha_input_prediction_loss'], dtype=tf.float32)
            return tf.multiply(
                alpha_tensor, tf.reduce_mean(tf.square(lin_system_input_pred - lin_system_input)),
                name='lin_input_prediction_loss')

    def l1_regularization_loss(self):
        """
        calculates l1 regularization of weights
        :return: l1 loss for weights
        """
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.params['alphal1'], scope=None)
        l1_loss = 0
        if len(self.nets.get_weights_state_encoder()) > 0:
            l1_loss = tf.contrib.layers.apply_regularization(
                l1_regularizer, weights_list=self.nets.get_weights_state_encoder())
        if len(self.nets.get_weights_input_encoder()) > 0:
            l1_loss += tf.contrib.layers.apply_regularization(
                l1_regularizer, weights_list=self.nets.get_weights_input_encoder())
        if len(self.nets.get_weights_state_decoder()) > 0:
            l1_loss += tf.contrib.layers.apply_regularization(
                l1_regularizer, weights_list=self.nets.get_weights_state_decoder())
        if len(self.nets.get_weights_input_decoder()) > 0:
            l1_loss += tf.contrib.layers.apply_regularization(
                l1_regularizer, weights_list=self.nets.get_weights_input_decoder())
        l1_loss += tf.contrib.layers.apply_regularization(l1_regularizer,
                                                          weights_list=[self.nets.get_lin_system_matrix()])
        l1_loss += tf.contrib.layers.apply_regularization(l1_regularizer,
                                                          weights_list=[self.nets.get_lin_input_vector()])
        return l1_loss

    def l2_regularization_loss(self):
        """
        calculates l2 regularization of weights
        :return: l2 loss for weights
        """
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.params['alphal2'], scope=None)
        l2_loss = 0
        if len(self.nets.get_weights_state_encoder()) > 0:
            l2_loss = tf.contrib.layers.apply_regularization(
                l2_regularizer, weights_list=self.nets.get_weights_state_encoder())
        if len(self.nets.get_weights_input_encoder()) > 0:
            l2_loss += tf.contrib.layers.apply_regularization(
                l2_regularizer, weights_list=self.nets.get_weights_input_encoder())
        if len(self.nets.get_weights_state_decoder()) > 0:
            l2_loss += tf.contrib.layers.apply_regularization(
                l2_regularizer, weights_list=self.nets.get_weights_state_decoder())
        if len(self.nets.get_weights_input_decoder()) > 0:
            l2_loss += tf.contrib.layers.apply_regularization(
                l2_regularizer, weights_list=self.nets.get_weights_input_decoder())
        l2_loss += tf.contrib.layers.apply_regularization(l2_regularizer,
                                                          weights_list=[self.nets.get_lin_system_matrix()])
        l2_loss += tf.contrib.layers.apply_regularization(l2_regularizer,
                                                          weights_list=[self.nets.get_lin_input_vector()])
        return l2_loss

    def state_distance_loss(self, state, lin_state):
        """
        calculates dhe distance loss
        :param state:
        :param lin_state:
        :return:
        """
        dist_lin_state = tf.reduce_mean(tf.square(lin_state), axis=2)
        Q = tf.constant(self.params['Q_matrix'], dtype=tf.float32)
        # outer matmul will lead to indipendend vectors to get multiplied.
        # multiply does element wise multiplikation and together with rreduce mean this will realize quadratic term
        dist_state = tf.reduce_mean(tf.multiply(tf.matmul(state, Q), state), axis=2)
        loss = tf.reduce_mean(tf.square(dist_lin_state - dist_state))
        alpha_tensor = tf.constant(self.params['alpha_state_distance_loss'], dtype=tf.float32)
        return tf.multiply(alpha_tensor, loss)

    def input_distance_loss(self, system_input, lin_system_input):
        """
        calculates dhe distance loss
        :param system_input:
        :param lin_system_input:
        :return:
        """
        '''
        dist_lin_input = tf.reduce_mean(tf.square(lin_system_input), axis=2)
        R = tf.constant(self.params['R_matrix'], dtype=tf.float32)
        dist_input = tf.reduce_mean(tf.multiply(tf.matmul(system_input, R), system_input), axis=2)
        loss = tf.reduce_mean(tf.square(dist_lin_input - dist_input))
        '''
        loss = tf.reduce_mean(tf.square(system_input - lin_system_input))
        alpha_tensor = tf.constant(self.params['alpha_input_distance_loss'], dtype=tf.float32)

        return tf.multiply(alpha_tensor, loss)

    def loss(self, state, system_input):
        """
        calculates the overall loss of the system
        :param state:
        :param system_input:
        :return: loss tensor
        """
        lin_state_pred, state_pred, lin_system_input_pred, system_input_pred = \
            self.nets.get_prediction(state, system_input)
        lin_state, lin_system_input = self.nets.transform_to_subspace(state, system_input)
        state_decoded = self.nets.get_state_decoder(lin_state[:, max(self.starting_point - 1, 0):, :])
        print(state_decoded.shape)
        state_prediction_loss = self.state_prediction_loss(state_pred, state)
        state_encoder_loss = self.state_encoder_loss(state_decoded, state)
        input_prediction_loss = self.input_prediction_loss(system_input_pred, system_input)
        lin_state_prediction_loss = self.lin_state_prediction_loss(lin_state_pred, lin_state)
        lin_input_prediction_loss = self.lin_input_prediction_loss(lin_system_input_pred, lin_system_input)
        dmd_loss = self.dmd_loss(lin_state, lin_system_input)

        l1_reg_tensor = self.l1_regularization_loss()
        l2_reg_tensor = self.l2_regularization_loss()

        total_loss = 0
        if self.params['choose_state_prediction_loss']:
            total_loss = state_prediction_loss
        if self.params['choose_state_encoder_loss']:
            total_loss += state_encoder_loss
        if self.params['choose_input_prediction_loss']:
            total_loss += input_prediction_loss
        if self.params['choose_lin_state_prediction_loss']:
            total_loss += lin_state_prediction_loss
        if self.params['choose_lin_input_prediction_loss']:
            total_loss += lin_input_prediction_loss
        if self.params['choose_dmd_loss']:
            total_loss += dmd_loss
        if self.params['choose_l1_reg']:
            total_loss += l1_reg_tensor
        if self.params['choose_l2_reg']:
            total_loss += l2_reg_tensor
        if 'choose_dist_loss' in self.params:
            if self.params['choose_dist_loss']:
                #dist_state = self.state_distance_loss(state, lin_state)
                dist_input = self.input_distance_loss(system_input, lin_system_input)
                #total_loss += dist_state + dist_input
                total_loss += dist_input

        return tf.identity(total_loss, name='total_loss'), state_prediction_loss, input_prediction_loss, \
            lin_state_prediction_loss, lin_input_prediction_loss, state_encoder_loss

