"""
this file implements a loader for model files to reload saved models
"""
import tensorflow as tf
import DKLc.networkarch as net
import numpy as np


class NetworkLoader:

    def __init__(self, filename, session, params, load_tf_graph=True):
        """
        constructor of class NetworkLoader
        :param filename: filename of the .meta file to load (without .meta)
        :param session: tensorflow session to load the model into
        """
        if load_tf_graph:
            self.filename = filename
            self.sess = session
            self.new_saver = tf.train.import_meta_graph(filename+'.meta')
            self.graph = tf.get_default_graph()
            self.new_saver.restore(self.sess, filename)
            self.total_loss = self.graph.get_tensor_by_name('total_loss:0')
            self.state_prediction_loss = self.graph.get_tensor_by_name('state_prediction_loss:0')
            self.input_prediction_loss = self.graph.get_tensor_by_name('input_prediction_loss:0')
            self.lin_state_prediction_loss = self.graph.get_tensor_by_name('lin_state_prediction_loss:0')
            self.lin_input_prediction_loss = self.graph.get_tensor_by_name('lin_input_prediction_loss:0')
            self.lin_state = self.graph.get_tensor_by_name('lin_state:0')
            self.lin_system_input = self.graph.get_tensor_by_name('lin_system_input:0')
            self.K = self.graph.get_tensor_by_name('K:0')
            self.L = self.graph.get_tensor_by_name('L:0')
            self.params = params

            # get weights and biases for nets
            self.weights_state_encoder = self.get_weights('state_encoder')
            self.weights_input_encoder = self.get_weights('input_encoder')
            self.weights_state_decoder = self.get_weights('state_decoder')
            self.weights_input_decoder = self.get_weights('input_decoder')

            self.biases_state_encoder = self.get_biases('state_encoder')
            self.biases_input_encoder = self.get_biases('input_encoder')
            self.biases_state_decoder = self.get_biases('state_decoder')
            self.biases_input_decoder = self.get_biases('input_decoder')


            # init nets and set weights (important dont init weights this will override loaded weights)
            self.state_encoder = net.NeuralNetwork(params['widths_state_encoder'], 'state_encoder', 'elu',
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'], init_weights=False)
            self.state_encoder.set_weights(self.weights_state_encoder)
            self.state_encoder.set_biases(self.biases_state_encoder)
            self.input_encoder = net.NeuralNetwork(params['widths_input_encoder'], 'input_encoder', 'elu',
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'], init_weights=False)
            self.input_encoder.set_weights(self.weights_input_encoder)
            self.input_encoder.set_biases(self.biases_input_encoder)
            self.state_decoder = net.NeuralNetwork(params['widths_state_decoder'], 'state_decoder', 'elu',
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'], init_weights=False)
            self.state_decoder.set_weights(self.weights_state_decoder)
            self.state_decoder.set_biases(self.biases_state_decoder)
            self.input_decoder = net.NeuralNetwork(params['widths_input_decoder'], 'input_decoder', 'elu',
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'], init_weights=False)
            self.input_decoder.set_weights(self.weights_input_decoder)
            self.input_decoder.set_biases(self.biases_input_decoder)

            if params['choose_sin_decoder']:
                weight_sin = self.graph.get_tensor_by_name('WeightSinstate_decoder:0')
                self.state_decoder.set_sin_weight(weight_sin)
                bias_sin = self.graph.get_tensor_by_name('BiasSinstate_decoder:0')
                self.state_decoder.set_sin_bias(bias_sin)

        else:
            self.filename = filename
            self.sess = session
            self.params = params

            activation_function = 'elu'
            if 'activation_function' in params:
                activation_function = params['activation_function']

            # init nets and set weights (important dont init weights this will override loaded weights)
            self.state_encoder = net.NeuralNetwork(params['widths_state_encoder'], 'state_encoder', activation_function,
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'],
                                                   initial_values_filename=filename+'state_encoder')
            self.input_encoder = net.NeuralNetwork(params['widths_input_encoder'], 'input_encoder', activation_function,
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'],
                                                   initial_values_filename=filename+'input_encoder')
            self.state_decoder = net.NeuralNetwork(params['widths_state_decoder'], 'state_decoder', activation_function,
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'],
                                                   initial_values_filename=filename+'state_decoder')
            self.input_decoder = net.NeuralNetwork(params['widths_input_decoder'], 'input_decoder', activation_function,
                                                   choose_dropout=params['choose_dropout'],
                                                   output_information=params['output_information'],
                                                   initial_values_filename=filename+'input_decoder')
            k_temp = np.loadtxt(filename+'K.csv', dtype=np.float32, ndmin=2)
            self.K_numeric = k_temp
            self.K = net.bias_variable(k_temp.shape, var_name='K', initial_value=k_temp)
            l_temp = np.loadtxt(filename+'L.csv', dtype=np.float32, ndmin=2)
            self.L_numeric = l_temp
            if l_temp.ndim == 1:
                l_temp = np.array([l_temp])
            self.L = net.bias_variable(k_temp.shape, var_name='L', initial_value=l_temp)

            if 'delay_coordinates_delay' not in self.params:
                self.params['delay_coordinates_delay'] = 0

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def predict_states(self, states, system_inputs):
        """
        predicts states
        :param states: tensor of one batch of reference trajecotry
        :param system_inputs: tensor one batch of input trajectory
        :return: predict states tensor
        """
        if self.params['concept'] == 1:
            y0 = 0
            if 'use_input_time_delay' in self.params:
                if self.params['use_input_time_delay']:
                    state_lin_part = 0
                    if len(states.shape) == 2:
                        begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                        _, end = system_inputs.shape
                        state_lin_part = self.state_encoder.get_net(
                            tf.concat([states[0, :], system_inputs[0, begin:end]], axis=0))
                    else:
                        begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                        _, _, end = system_inputs.shape
                        state_lin_part = \
                            self.state_encoder.get_net(tf.concat([states[:, 0, :], system_inputs[:, 0, begin:end]], axis=1))
                    y0 = state_lin_part
                else:
                    y0 = self.state_encoder.get_net(states[:, 0, :])
            else:
                y0 = self.state_encoder.get_net(states[:, 0, :])
            y_list = [y0]
            lin_system_inputs = 0
            if self.params['use_input_time_delay']:
                lin_system_inputs = self.input_encoder.get_net(
                    system_inputs[
                    :, :, 0:int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))])
            else:
                if not self.params['is_input_affine']:
                    lin_system_inputs = self.input_encoder.get_net(tf.concat([states, system_inputs], axis=2))
                else:
                    lin_system_inputs = self.input_encoder.get_net(system_inputs)

            for i in range(1, self.params['prediction_horizon'] + 1):
                y_list.append(tf.matmul(y_list[i - 1], self.K) + tf.matmul(lin_system_inputs[:, i - 1, :], self.L))
            lin_states = tf.stack(y_list, axis=1)
            predicted_states = self.state_decoder.get_net(lin_states)
            lin_states_ref = 0

            if 'use_input_time_delay' in self.params:
                if self.params['use_input_time_delay']:
                    state_lin_part = 0
                    if len(states.shape) == 2:
                        begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                        _, end = system_inputs.shape
                        state_lin_part = self.state_encoder.get_net(
                            tf.concat([states[:, :], system_inputs[:, begin:end]], axis=1))
                    else:
                        begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                        _, _, end = system_inputs.shape
                        state_lin_part = \
                            self.state_encoder.get_net(
                                tf.concat([states[:, :, :], system_inputs[:, :, begin:end]], axis=2))
                    lin_states_ref = state_lin_part
                else:
                    lin_states_ref = self.state_encoder.get_net(states)
            else:
                lin_states_ref = self.state_encoder.get_net(states)
            return predicted_states, lin_states, lin_states_ref
        elif self.params['concept'] == 2:
            y0 = 0
            if 'use_input_time_delay' in self.params:
                begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                if self.params['use_input_time_delay']:
                    state_lin_part = 0
                    if len(states.shape) == 2:
                        _, end = system_inputs.shape
                        state_lin_part = self.state_encoder.get_net(
                            tf.concat([states, system_inputs[0, begin:end]], axis=0))
                    else:
                        _, _, end = system_inputs.shape
                        state_lin_part = \
                            self.state_encoder.get_net(
                                tf.concat([states[:, 0, :], system_inputs[:, 0, begin:end]], axis=1))
                    begin = int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))
                    y0 = tf.concat([states[:, 0, 0:begin], state_lin_part], axis=1)
                else:
                    begin = int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))
                    y0 = tf.concat([states[:, 0, 0:begin], self.state_encoder.get_net(states[:, 0, :])], axis=1)
            else:
                y0 = tf.concat([states[:, 0, :], self.state_encoder.get_net(states[:, 0, :])], axis=1)
            y_list = [y0]
            print(y0.shape)
            lin_system_inputs = 0
            if self.params['use_input_time_delay']:
                lin_system_inputs = self.input_encoder.get_net(
                    system_inputs[
                    :, :, 0:int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))])
            else:
                if not self.params['is_input_affine']:
                    lin_system_inputs = self.input_encoder.get_net(tf.concat([states, system_inputs], axis=2))
                else:
                    lin_system_inputs = self.input_encoder.get_net(system_inputs)

            for i in range(1, self.params['prediction_horizon'] + 1):
                y_list.append(tf.matmul(y_list[i - 1], self.K) + tf.matmul(lin_system_inputs[:, i - 1, :], self.L))
            lin_states = tf.stack(y_list, axis=1)
            lin_states_ref = 0

            if 'use_input_time_delay' in self.params:
                if self.params['use_input_time_delay']:
                    state_lin_part = 0
                    begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                    if len(states.shape) == 2:
                        _, end = system_inputs.shape
                        state_lin_part = self.state_encoder.get_net(
                            tf.concat([states, system_inputs[:, begin:end]], axis=1))
                        begin = int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))
                        lin_states_ref = tf.concat([states[:, 0:begin], state_lin_part], axis=len(states.shape) - 1)
                    else:
                        _, _, end = system_inputs.shape
                        state_lin_part = \
                            self.state_encoder.get_net(tf.concat([states, system_inputs[:, :, begin:end]], axis=2))
                        begin = int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))
                        lin_states_ref = tf.concat([states[:, :, 0:begin], state_lin_part], axis=len(states.shape) - 1)
                else:
                    lin_states_ref = tf.concat([states[:, :, :], self.state_encoder.get_net(states)], axis=2)
            else:
                lin_states_ref = tf.concat([states[:, :, :], self.state_encoder.get_net(states)], axis=2)
            return lin_states, lin_states_ref

    def predict_states_nonlinear(self, states, system_inputs):
        """
        predicts states
        :param states: tensor of one batch of reference trajecotry
        :param system_inputs: tensor one batch of input trajectory
        :return: predict states tensor
        """
        y0 = self.state_encoder.get_net(states[:, 0, :])
        y_list = [y0]
        x_pred_list = [self.state_decoder.get_net(y0)]
        lin_system_inputs = 0
        if not self.params['is_input_affine']:
            lin_system_inputs = self.input_encoder.get_net(tf.concat([states, system_inputs], axis=2))
        else:
            lin_system_inputs = self.input_encoder.get_net(system_inputs)

        for i in range(1, self.params['prediction_horizon'] + 1):
            y_list.append(tf.matmul(self.state_encoder.get_net(x_pred_list[i - 1]), self.K) +
                          tf.matmul(lin_system_inputs[:, i - 1, :], self.L))
            x_pred_list.append(self.get_state_decoder(y_list[i]))
        lin_states = tf.stack(y_list, axis=1)
        predicted_states = tf.stack(x_pred_list, axis=1)
        print(predicted_states.shape)
        return predicted_states, lin_states

    def get_weights(self, name):
        weights = []
        for i in range(0, len(self.params['widths_' + name]) - 1):
            weights.append(self.graph.get_tensor_by_name('Weight' + name + str(i) + ':0'))
        return weights

    def get_biases(self, name):
        biases = []
        for i in range(0, len(self.params['widths_' + name]) - 1):
            biases.append(self.graph.get_tensor_by_name('Bias' + name + str(i) + ':0'))
        return biases

    def get_state_encoder(self, state, system_input=None):
        """
        :param state: state[i,:] =  state_i (state_i is a state of batch element i)
        :param system_input:
        :return: tensor of state encoder net with input tensor  state
        """
        if 'use_input_time_delay' in self.params:
            if self.params['use_input_time_delay']:
                state_lin_part = 0
                # concat state and system input to be the input for the neural network
                if len(state.shape) == 2:
                    # only use last inputs and not the current one
                    begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                    _, end = system_input.shape
                    state_lin_part = self.state_encoder.get_net(tf.concat([state, system_input[:, begin:end]], axis=1))
                    if self.params['concept'] == 1:
                        return state_lin_part
                    if self.params['concept'] == 2:
                        begin = int(self.params['dim_system_states'] / (self.params['delay_coordinates_delay'] + 1))
                        return tf.concat([state[:, 0:begin], state_lin_part], axis=len(state.shape) - 1)
                        #return tf.concat([state[:, :], state_lin_part], axis=len(state.shape) - 1)
                    else:
                        ValueError('Selected concept does not exist')
                else:
                    # only use last inputs and not the current one
                    begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
                    _, _, end = system_input.shape
                    state_lin_part = \
                        self.state_encoder.get_net(tf.concat([state, system_input[:, :, begin:end]], axis=2))
                    if self.params['concept'] == 1:
                        return state_lin_part
                    if self.params['concept'] == 2:
                        begin = int(self.params['dim_system_states'] / (self.params['delay_coordinates_delay'] + 1))
                        return tf.concat([state[:, :, 0:begin], state_lin_part], axis=len(state.shape) - 1)
                    else:
                        ValueError('Selected concept does not exist')

        if self.params['concept'] == 1:
            return self.state_encoder.get_net(state)
        if self.params['concept'] == 2:
            return tf.concat([state, self.state_encoder.get_net(state)], axis=len(state.shape) - 1)
        else:
            ValueError('Selected concept does not exist')

    def get_state_decoder(self, lin_state):
        """
        same as get_state_encoder() just for decoder
        :param lin_state:
        :return:
        """
        if self.params['concept'] == 1:
            return self.state_decoder.get_net(lin_state)
        if self.params['concept'] == 2:
            if len(lin_state.shape) == 2:
                return tf.identity(
                    lin_state[:, 0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))])
            elif len(lin_state.shape) == 3:
                return tf.identity(
                    lin_state[:, :,
                    0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))])
            else:
                ValueError('Wrong lin state rank')
        else:
            ValueError('Selected concept does not exist')

