'''
This file contains all necessary classes for building the tensors of the different nets used
'''

import tensorflow as tf
import numpy as np
import os.path


def weight_variable(shape, var_name, distribution='tn', scale=0.2, initial_value=None, initialize_zero=False):
    """Create a variable for a weight matrix. from https://github.com/BethanyL/DeepKoopman

    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization (default 'tn')
        scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)

    Returns:
        a TensorFlow variable for a weight matrix

    Side effects:
        None

    Raises ValueError if distribution is filename but shape of data in file does not match input shape
    """
    if initial_value is not None:
        return tf.Variable(initial_value, name=var_name, trainable=True)
    if initialize_zero:
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32), name=var_name, trainable=True)
    if distribution == 'tn':
        initial = tf.truncated_normal(shape, stddev=scale, dtype=tf.float32)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float32)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float32)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name, trainable=True)


def bias_variable(shape, var_name, distribution='', initial_value=None):
    """Create a variable for a bias vector. from https://github.com/BethanyL/DeepKoopman

    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')

    Returns:
        a TensorFlow variable for a bias vector

    Side effects:
        None
    """
    if initial_value is not None:
        return tf.Variable(initial_value, name=var_name, trainable=True)
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float32)
    else:
        initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=var_name, trainable=True)


class NeuralNetwork:

    def __init__(self, widths, name, activation, choose_dropout=True, dropout_rate=0.1, output_information=False,
                 initial_values_filename=None, init_weights=True, initialize_zero=False):
        """
        constructor of class NeuralNetwork
        :param widths: array with the widths of the layers
        :param name: name of the net
        :param activation: type of activation function
        :param choose_dropout: use dropout layers
        :param dropout_rate:
        :param output_information: output information on console
        :param init_weights: init the weights (if you do not want to train but reload old data you can set this to
        0 to avoid init the weights)
        """
        self.widths = widths
        self.name = name
        self.activation = activation
        self.choose_dropout = choose_dropout
        self.dropout_rate = dropout_rate
        self.output_information = output_information
        self.initialize_zero = initialize_zero

        self.weights = None
        self.biases = None
        if init_weights:
            if initial_values_filename is None:
                self.init_weights_biases()
            else:
                self.load_net(initial_values_filename)

        self.sin_weight = 0
        self.sin_bias = 0
        if self.activation == 'sin':
            self.sin_weight = weight_variable((self.widths[0], self.widths[1]), 'WeightSin'+self.name)
            self.sin_bias = bias_variable((self.widths[1],), 'BiasSin'+self.name)

    def save_net(self, path, sess):
        """
        writes weights of every layer into a seperate file
        :param path:
        :param sess:
        :return:
        """
        for i in range(0, len(self.weights)):
            np.savetxt(path+self.name+'WeightsLayer'+str(i)+'.csv', sess.run(self.weights[i]))
        for i in range(0, len(self.biases)):
            np.savetxt(path+self.name+'BiasesLayer'+str(i)+'.csv', sess.run(self.biases[i]))

    def load_net(self, filename):
        """
        loads weights and biases from file
        :param filename: files should have the form filename<Weights/Biases>Layer<nmbr>.csv
        :return:
        """
        i = 0
        self.weights = []
        while os.path.isfile(filename+'WeightsLayer'+str(i)+'.csv'):
            weight = np.loadtxt(filename+'WeightsLayer'+str(i)+'.csv', dtype=np.float32, ndmin=2)
            self.weights.append(weight_variable(weight.shape, 'Weight'+self.name+str(i), initial_value=weight))
            self.console_log(self.name + ' Layer ' + str(i) + ' has ' + str(self.weights[i].shape) + ' weights')
            i += 1

        i = 0
        self.biases = []
        while os.path.isfile(filename+'BiasesLayer'+str(i)+'.csv'):
            bias = np.loadtxt(filename+'BiasesLayer'+str(i)+'.csv', dtype=np.float32, ndmin=1)
            self.biases.append(weight_variable(bias.shape, 'Bias'+self.name+str(i), initial_value=bias))
            self.console_log(self.name + ' Layer ' + str(i) + ' has ' + str(self.biases[i].shape) + ' biases')
            i += 1

    def console_log(self, message):
        if self.output_information:
            print(message)

    def init_weights_biases(self):
        """
        initializes weights. (motivated out of https://github.com/BethanyL/DeepKoopman)
        :return:
        """
        self.weights = []
        for i in range(0, len(self.widths) - 1):
            self.weights.append(weight_variable((self.widths[i], self.widths[i+1]), 'Weight'+self.name+str(i),
                                                initialize_zero=self.initialize_zero))
            self.console_log(self.name + ' Layer ' + str(i) + ' has ' + str(self.weights[i].shape) + ' weights')

        self.console_log('Successfully initialized weights of ' + self.name)

        self.biases = []
        for i in range(0, len(self.widths) - 1):
            self.biases.append(bias_variable((self.widths[i + 1],), 'Bias'+self.name+str(i)))
            self.console_log(self.name + ' Layer ' + str(i) + ' has ' + str(self.biases[i].shape) + ' biases')

        self.console_log('Successfully initialized biases of ' + self.name + '\n')

    def get_net(self, input_tensor):
        """
        returns the output tensor with input as input tensor (from https://github.com/BethanyL/DeepKoopman).
        if length of weights equals zero the output tensor will be the input tensor.
        :param input_tensor:
        :return: output tensor of net
        """
        prev_layer = input_tensor
        prev_layer_sin = input_tensor
        for i in range(0, len(self.weights) - 1):
            prev_layer = tf.matmul(prev_layer, self.weights[i]) + self.biases[i]
            if self.activation == 'sin' and i == 0:
                prev_layer_sin = tf.matmul(prev_layer_sin, self.sin_weight) + self.sin_bias
            if self.activation == 'sigmoid':
                prev_layer = tf.sigmoid(prev_layer)
            elif self.activation == 'relu':
                prev_layer = tf.nn.relu(prev_layer)
            elif self.activation == 'elu':
                prev_layer = tf.nn.elu(prev_layer)
            elif self.activation == 'sin':
                if i == 0:
                    # add sin activations in first layer
                    prev_layer = tf.nn.elu(prev_layer) + tf.math.sin(prev_layer_sin)
                else:
                    prev_layer = tf.nn.elu(prev_layer)

            # do not apply dropout to the first and last layer as dropout in this place will have a significant
            # and thus a prob. negative influence (because it will block an entire input or output)
            if self.choose_dropout:
                prev_layer = tf.nn.dropout(prev_layer, rate=self.dropout_rate)

        # apply last layer without any nonlinearity
        if len(self.weights) > 0:
            prev_layer = tf.matmul(
                prev_layer, self.weights[len(self.weights) - 1]) + self.biases[len(self.weights) - 1]

        return prev_layer

    def get_trainable_vars(self):
        """
        :return: al trainable variables (weights and biases)
        """
        return self.weights + self.biases

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_sin_weight(self):
        return self.sin_weight

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases

    def set_sin_weight(self, sin_weight):
        self.sin_weight = sin_weight

    def set_sin_bias(self, sin_bias):
        self.sin_bias = sin_bias


class RecurrentNeuralNetwork(NeuralNetwork):
    def __init__(self, widths, name, activation, pos_lstm_layer,
                 choose_dropout=True, dropout_rate=0.1, output_information=False,
                 initial_values_filename=None, init_weights=True, initialize_zero=False):
        """
        constructor of class NeuralNetwork
        :param widths: array with the widths of the layers
        :param name: name of the net
        :param activation: type of activation function
        :param choose_dropout: use dropout layers
        :param dropout_rate:
        :param output_information: output information on console
        :param init_weights: init the weights (if you do not want to train but reload old data you can set this to
        0 to avoid init the weights)
        """
        self.pos_lstm_layer = pos_lstm_layer

        # w0 is width at position of rnn layer
        self.u0 = 0
        self.v0 = 0
        self.b0 = 0

        self.ui = 0
        self.wi = 0
        self.bi = 0

        self.uf = 0
        self.wf = 0
        self.bf = 0

        self.uk = 0
        self.wk = 0
        self.bk = 0

        super().__init__(widths, name, activation, choose_dropout=choose_dropout, dropout_rate=dropout_rate,
                         output_information=output_information, initial_values_filename=initial_values_filename,
                         init_weights=init_weights, initialize_zero=initialize_zero)

        if initial_values_filename is None:
            self.init_lstm_weights()

    def init_lstm_weights(self):
        dim_state = self.widths[self.pos_lstm_layer]
        dim_input = self.widths[self.pos_lstm_layer - 1]
        self.u0 = weight_variable((dim_state, dim_state), var_name='u0')
        self.v0 = weight_variable((dim_state, dim_state), var_name='v0')
        self.b0 = bias_variable((dim_state,), var_name='b0')

        self.ui = weight_variable((dim_state, dim_state), var_name='ui')
        self.wi = weight_variable((dim_input, dim_state), var_name='wi')
        self.bi = bias_variable((dim_state,), var_name='bi')

        self.uf = weight_variable((dim_state, dim_state), var_name='uf')
        self.wf = weight_variable((dim_input, dim_state), var_name='wf')
        self.bf = bias_variable((dim_state,), var_name='bf')

        self.uk = weight_variable((dim_state, dim_state), var_name='uk')
        self.wk = weight_variable((dim_input, dim_state), var_name='wk')
        self.bk = bias_variable((dim_state,), var_name='bk')

    def save_net(self, path, sess):
        # save net weights
        super().save_net(path, sess)

        # save lstm weights
        np.savetxt(path + self.name + 'u0.csv', sess.run(self.u0))
        np.savetxt(path + self.name + 'v0.csv', sess.run(self.v0))
        np.savetxt(path + self.name + 'b0.csv', sess.run(self.b0))

        np.savetxt(path + self.name + 'ui.csv', sess.run(self.ui))
        np.savetxt(path + self.name + 'wi.csv', sess.run(self.wi))
        np.savetxt(path + self.name + 'bi.csv', sess.run(self.bi))

        np.savetxt(path + self.name + 'uf.csv', sess.run(self.uf))
        np.savetxt(path + self.name + 'wf.csv', sess.run(self.wf))
        np.savetxt(path + self.name + 'bf.csv', sess.run(self.bf))

        np.savetxt(path + self.name + 'uk.csv', sess.run(self.uk))
        np.savetxt(path + self.name + 'wk.csv', sess.run(self.wk))
        np.savetxt(path + self.name + 'bk.csv', sess.run(self.bk))

    def load_net(self, filename):
        super().load_net(filename)

        temp = np.loadtxt(filename + 'u0.csv', dtype=np.float32, ndmin=2)
        self.u0 = weight_variable(temp.shape, 'u0', initial_value=temp)

        temp = np.loadtxt(filename + 'v0.csv', dtype=np.float32, ndmin=2)
        self.v0 = weight_variable(temp.shape, 'v0', initial_value=temp)

        temp = np.loadtxt(filename + 'b0.csv', dtype=np.float32, ndmin=1)
        self.b0 = weight_variable(temp.shape, 'b0', initial_value=temp)

        # i
        temp = np.loadtxt(filename + 'ui.csv', dtype=np.float32, ndmin=2)
        self.ui = weight_variable(temp.shape, 'ui', initial_value=temp)

        temp = np.loadtxt(filename + 'wi.csv', dtype=np.float32, ndmin=2)
        self.wi = weight_variable(temp.shape, 'wi', initial_value=temp)

        temp = np.loadtxt(filename + 'bi.csv', dtype=np.float32, ndmin=1)
        self.bi = weight_variable(temp.shape, 'bi', initial_value=temp)

        # f
        temp = np.loadtxt(filename + 'uf.csv', dtype=np.float32, ndmin=2)
        self.uf = weight_variable(temp.shape, 'uf', initial_value=temp)

        temp = np.loadtxt(filename + 'wf.csv', dtype=np.float32, ndmin=2)
        self.wf = weight_variable(temp.shape, 'wf', initial_value=temp)

        temp = np.loadtxt(filename + 'bf.csv', dtype=np.float32, ndmin=1)
        self.bf = weight_variable(temp.shape, 'bf', initial_value=temp)

        # k
        temp = np.loadtxt(filename + 'uk.csv', dtype=np.float32, ndmin=2)
        self.uk = weight_variable(temp.shape, 'uk', initial_value=temp)

        temp = np.loadtxt(filename + 'wk.csv', dtype=np.float32, ndmin=2)
        self.wk = weight_variable(temp.shape, 'wk', initial_value=temp)

        temp = np.loadtxt(filename + 'bk.csv', dtype=np.float32, ndmin=1)
        self.bk = weight_variable(temp.shape, 'bk', initial_value=temp)

    def get_net(self, input_tensor):
        """
        returns the output tensor with input as input tensor (from https://github.com/BethanyL/DeepKoopman).
        if length of weights equals zero the output tensor will be the input tensor.
        :param input_tensor:
        :return: output tensor of net
        """
        prev_layer = input_tensor
        prev_layer_sin = input_tensor
        for i in range(0, len(self.weights) - 1):
            lstm_input = prev_layer
            prev_layer = tf.matmul(prev_layer, self.weights[i]) + self.biases[i]
            if i != self.pos_lstm_layer - 1:
                if self.activation == 'sin' and i == 0:
                    prev_layer_sin = tf.matmul(prev_layer_sin, self.sin_weight) + self.sin_bias
                if self.activation == 'sigmoid':
                    prev_layer = tf.sigmoid(prev_layer)
                elif self.activation == 'relu':
                    prev_layer = tf.nn.relu(prev_layer)
                elif self.activation == 'elu':
                    prev_layer = tf.nn.elu(prev_layer)
                elif self.activation == 'sin':
                    if i == 0:
                        # add sin activations in first layer
                        prev_layer = tf.nn.elu(prev_layer) + tf.math.sin(prev_layer_sin)
                    else:
                        prev_layer = tf.nn.elu(prev_layer)
            else:
                if len(input_tensor.shape) == 3:
                    batch_size, traj_length, _ = input_tensor.shape
                    state = tf.zeros((1, self.widths[self.pos_lstm_layer]), dtype=tf.float32)
                    ct = tf.zeros((1, self.widths[self.pos_lstm_layer]), dtype=tf.float32)
                    output_list = []
                    for j in range(0, traj_length):
                        # calculate output
                        # prev layer weight *input + b0
                        output = tf.nn.tanh(tf.matmul(state, self.u0) + tf.matmul(ct, self.v0) + prev_layer[:, j, :])

                        # calculate ct
                        it = tf.nn.tanh(tf.matmul(state, self.ui) + tf.matmul(lstm_input[:, j, :], self.wi) + self.bi)
                        ft = tf.nn.tanh(tf.matmul(state, self.uf) + tf.matmul(lstm_input[:, j, :], self.wf) + self.bf)
                        kt = tf.nn.tanh(tf.matmul(state, self.uk) + tf.matmul(lstm_input[:, j, :], self.wk) + self.bk)
                        ct = tf.multiply(it, kt) + tf.multiply(ct, ft)

                        output_list.append(output)
                        state = output
                    prev_layer = tf.stack(output_list, axis=1)
                else:
                    raise ValueError('input tensor has not dimension 3')

            # do not apply dropout to the first and last layer as dropout in this place will have a significant
            # and thus a prob. negative influence (because it will block an entire input or output)
            if self.choose_dropout:
                prev_layer = tf.nn.dropout(prev_layer, rate=self.dropout_rate)

        # apply last layer without any nonlinearity
        if len(self.weights) > 0:
            prev_layer = tf.matmul(
                prev_layer, self.weights[len(self.weights) - 1]) + self.biases[len(self.weights) - 1]

        return prev_layer


class Nets:
    def __init__(self, params, state_encoder_file=None, input_encoder_file=None, state_decoder_file=None,
                 input_decoder_file=None,
                 state_matrix_filename=None, input_vector_filename=None):
        self.params = params
        activation_function = 'elu'
        if 'activation_function' in params:
            activation_function = params['activation_function']
        activation_decoder = activation_function
        if params['choose_sin_decoder']:
            activation_decoder = 'sin'
        self.use_lstm = False
        if 'use_lstm_layer' in params:
            self.use_lstm = params['use_lstm_layer']

        if not self.use_lstm:
            self.state_encoder = NeuralNetwork(params['widths_state_encoder'], 'state_encoder', activation_function,
                                               choose_dropout=params['choose_dropout'],
                                               dropout_rate=params['dropout_rate'],
                                               output_information=params['output_information'],
                                               initial_values_filename=state_encoder_file,
                                               initialize_zero=params['initialize_zero'])
        else:
            self.state_encoder = RecurrentNeuralNetwork(params['widths_state_encoder'], 'state_encoder',
                                                        activation_function,
                                                        choose_dropout=params['choose_dropout'],
                                                        dropout_rate=params['dropout_rate'],
                                                        output_information=params['output_information'],
                                                        initial_values_filename=state_encoder_file,
                                                        initialize_zero=params['initialize_zero'],
                                                        pos_lstm_layer=params['pos_lstm_layer'])

        self.state_decoder = NeuralNetwork(params['widths_state_decoder'], 'state_decoder', activation_decoder,
                                           choose_dropout=params['choose_dropout'],
                                           dropout_rate=params['dropout_rate'],
                                           output_information=params['output_information'],
                                           initial_values_filename=state_decoder_file,
                                           initialize_zero=params['initialize_zero'])

        self.input_encoder = NeuralNetwork(params['widths_input_encoder'], 'input_encoder', activation_function,
                                           choose_dropout=params['choose_dropout'],
                                           dropout_rate=params['dropout_rate'],
                                           output_information=params['output_information'],
                                           initial_values_filename=input_encoder_file,
                                           initialize_zero=params['initialize_zero'])

        self.input_decoder = NeuralNetwork(params['widths_input_decoder'], 'input_decoder', activation_function,
                                           choose_dropout=params['choose_dropout'],
                                           dropout_rate=params['dropout_rate'],
                                           output_information=params['output_information'],
                                           initial_values_filename=input_decoder_file,
                                           initialize_zero=params['initialize_zero'])

        self.use_input_time_delay = False
        if 'use_input_time_delay' in self.params:
            self.use_input_time_delay = self.params['use_input_time_delay']

        # state matrix !!!!!! the update for the state is y(k+1) = y(k)*K + u*L. So the normal state matrix is
        # transp(K)
        # initialize as bias (with 0) as initialization with random parameters will lead to nan output when
        # matrix has high dimension
        if state_matrix_filename is not None:
            k_temp = np.loadtxt(state_matrix_filename, dtype=np.float32, ndmin=2)
            self.K = weight_variable(k_temp.shape, var_name='K', initial_value=k_temp)
        else:
            self.K = weight_variable((params['dim_lin_system'], params['dim_lin_system']), var_name='K')
        if input_vector_filename is not None:
            l_temp = np.loadtxt(input_vector_filename, dtype=np.float32, ndmin=2)
            if l_temp.ndim == 1:
                l_temp = np.array([l_temp])
            self.L = weight_variable(l_temp.shape, var_name='L', initial_value=l_temp)
        else:
            self.L = weight_variable((params['widths_input_encoder'][len(params['widths_input_encoder']) - 1],
                                    params['dim_lin_system']), var_name='L')

        # for variance analysis this has to be set to be random
        if 'initialize_lin_model_random' in params:
            if params['initialize_lin_model_random']:
                self.K = weight_variable((params['dim_lin_system'], params['dim_lin_system']), var_name='K')
                self.L = weight_variable((params['widths_input_encoder'][len(params['widths_input_encoder']) - 1],
                                        params['dim_lin_system']), var_name='L')

        if self.params['choose_constant_system_matrix']:
            self.K = tf.Variable(params['constant_system_matrix'],
                                 name='K', trainable=False, dtype=tf.float32)

        self.stop = 0
        if 'rnn_start_length' in self.params and self.use_lstm:
            self.stop = self.params['rnn_start_length'] - 1

    def load_nets(self, path):
        self.state_encoder.load_net(str(path) + '/state_encoder')
        if self.params['concept'] == 1:
            self.state_decoder.load_net(str(path) + '/state_decoder')

        if len(self.params['widths_input_encoder']) == 1 and \
                self.params['widths_input_encoder'][0] == self.params['dim_system_input']:
            self.input_encoder.load_net(str(path) + '/input_encoder')
            self.input_decoder.load_net(str(path) + '/input_decoder')

        K_mat = np.loadtxt(str(path) + '/K.csv', dtype=np.float32, ndmin=2)
        self.K = bias_variable(K_mat.shape, var_name='K', initial_value=K_mat)
        L_mat = np.loadtxt(str(path) + '/L.csv', dtype=np.float32, ndmin=2)
        self.L = bias_variable(L_mat.shape, var_name='L', initial_value=L_mat)

    def save_nets(self, path, sess):
        """
        saves nets and linear model
        :param path: path to save to
        :param sess: current tensorflow session
        :return:
        """
        self.state_encoder.save_net(path, sess)
        self.input_encoder.save_net(path, sess)
        self.state_decoder.save_net(path, sess)
        self.input_decoder.save_net(path, sess)

        np.savetxt(path+'K.csv', sess.run(self.K))
        np.savetxt(path+'L.csv', sess.run(self.L))

    def get_prediction(self, state, system_input):
        """
        calculate prediction tensors of the system. the concept can be selected using params['concept']
        :param state: state tensor. state[j, i, :] contains state of batch element j of timepoint i
        :param system_input: input tensor. system_input[j, i, :] contains input of batch element j of timepoint i
        :return: y_pred: prediction of linear state,
                 x_pred: prediction of state,
                 v_pred: prediction of the linear input,
                 u_pred: prediction of the input
        """
        if self.params['concept'] == 1:
            lin_state_pred, state_pred, lin_system_input_pred, system_input_pred = \
                self.get_prediction_concept1(state, system_input)
            return lin_state_pred, state_pred, lin_system_input_pred, system_input_pred
        if self.params['concept'] == 2:
            lin_state_pred, state_pred, lin_system_input_pred, system_input_pred = \
                self.get_prediction_concept2(state, system_input)
            return lin_state_pred, state_pred, lin_system_input_pred, system_input_pred
        else:
            ValueError('Selected concept does not exist')

    def get_prediction_concept1(self, state, system_input):
        """
        calculate prediction tensors of the system for concept 1
        :param state: state tensor. state[j, i, :] contains state of batch element j of timepoint i
        :param system_input: input tensor. system_input[j, i, :] contains input of batch element j of timepoint i
        :return: y_pred: prediction of linear state,
                 x_pred: prediction of state,
                 v_pred: prediction of the linear input,
                 u_pred: prediction of the input
        """
        if not self.params['predict_input']:
            lin_system_input_pred = self.get_input_encoder_batch(state, system_input)
            system_input_pred = self.get_input_decoder_batch(state, lin_system_input_pred)

            # linear state at time 0
            y0 = 0
            if self.use_input_time_delay:
                y0 = self.get_state_encoder(state[:, 0, :], system_input[:, 0, :])
            elif self.use_lstm:
                # start rnn
                state_start = state[:, 0:self.params['rnn_start_length'], :]
                system_input_start = system_input[:, 0:self.params['rnn_start_length'], :]
                y = self.get_state_encoder(state_start, system_input_start)
                y0 = y[:, -1, :]
            else:
                y0 = self.get_state_encoder(state[:, 0, :])
            y_list = [y0]
            # loop over the whole time horizon and predict using linear model
            for i in range(1, self.params['prediction_horizon'] + 1 - self.stop):
                # perform linear model step
                yi = self.get_lin_model(y_list[i-1], lin_system_input_pred[:, i - 1 + self.stop, :])
                y_list.append(yi)

            # convert lists to 3D Tensors (time axis = 1)
            lin_state_pred = tf.stack(y_list, axis=1)
            state_pred = self.get_state_decoder(lin_state_pred)
            return lin_state_pred, state_pred, lin_system_input_pred, system_input_pred
        else:
            # lin_system_input_pred = self.get_input_encoder_batch(state, system_input)

            # linear state at time 0
            y0 = self.get_state_encoder(state[:, 0, :])
            y_list = [y0]
            v0 = self.get_input_encoder(state[:, 0, :], system_input[:, 0, :])
            v_list = [v0]

            x_pred0 = self.get_state_decoder(y0)
            x_pred_list = [x_pred0]
            # loop over the whole time horizon and predict using linear model
            for i in range(1, self.params['prediction_horizon'] + 1):
                # perform linear model step
                yi = self.get_lin_model(y_list[i - 1], v_list[i - 1])
                y_list.append(yi)

                x_pred_i = self.get_state_decoder(yi)
                x_pred_list.append(x_pred_i)

                v_i = self.get_input_encoder(x_pred_i, system_input[:, i, :])
                v_list.append(v_i)

            # convert lists to 3D Tensors (time axis = 1)
            lin_state_pred = tf.stack(y_list, axis=1)
            state_pred = tf.stack(x_pred_list, axis=1)
            lin_system_input_pred = tf.stack(v_list, axis=1)
            system_input_pred = self.get_input_decoder_batch(state, lin_system_input_pred)
            return lin_state_pred, state_pred, lin_system_input_pred, system_input_pred

    def get_prediction_concept2(self, state, system_input):
        """
        calculate prediction tensors of the system for concept 2
        :param state: state tensor. state[j, i, :] contains state of batch element j of timepoint i
        :param system_input: input tensor. system_input[j, i, :] contains input of batch element j of timepoint i
        :return: y_pred: prediction of linear state,
                 x_pred: prediction of state,
                 v_pred: prediction of the linear input,
                 u_pred: prediction of the input
        """
        lin_system_input_pred = self.get_input_encoder_batch(state, system_input)
        system_input_pred = self.get_input_decoder_batch(state, lin_system_input_pred)

        # linear state at time 0
        y0 = 0
        if self.use_input_time_delay:
            y0 = self.get_state_encoder(state[:, 0, :], system_input[:, 0, :])
        elif self.use_lstm:
            # start rnn
            state_start = state[:, 0:self.params['rnn_start_length'], :]
            system_input_start = system_input[:, 0:self.params['rnn_start_length'], :]
            y = self.get_state_encoder(state_start, system_input_start)
            y0 = y[:, -1, :]
        else:
            y0 = self.get_state_encoder(state[:, 0, :])
        y_list = [y0]
        # loop over the whole time horizon and predict using linear model
        for i in range(1, self.params['prediction_horizon'] + 1 - self.stop):
            # perform linear model step
            yi = self.get_lin_model(y_list[i-1], lin_system_input_pred[:, i - 1 + self.stop, :])
            y_list.append(yi)

        # convert lists to 3D Tensors (time axis = 1)
        lin_state_pred = tf.stack(y_list, axis=1)
        state_pred = self.get_state_decoder(lin_state_pred)
        return lin_state_pred, state_pred, lin_system_input_pred, system_input_pred

    def transform_to_subspace(self, state, system_input):
        """
        transform state and system input (both 3D tensors) with second axis equals the time axis to the linear subspace
        :param state:
        :param system_input:
        :return: linear state tensor and linear system input tensor
        """
        lin_state = 0
        if self.use_input_time_delay:
            lin_state = self.get_state_encoder(state, system_input)
        elif self.use_lstm:
            lin_state = self.get_state_encoder(state, system_input)
        else:
            lin_state = self.get_state_encoder(state)
        lin_system_input = self.get_input_encoder_batch(state, system_input)
        return tf.identity(lin_state, name='lin_state'), tf.identity(lin_system_input, name='lin_system_input')

    def get_state_encoder(self, state, system_input=None):
        """
        :param state: state[i,:] =  state_i (state_i is a state of batch element i)
        :param system_input: input of system (optional only for time delay)
        :return: tensor of state encoder net with input tensor  state
        """
        if self.use_input_time_delay:
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
                    begin = int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))
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
                    begin = int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))
                    return tf.concat([state[:, :, 0:begin], state_lin_part], axis=len(state.shape) - 1)
                    #return tf.concat([state[:, :, :], state_lin_part], axis=len(state.shape) - 1)
                else:
                    ValueError('Selected concept does not exist')

        if self.params['concept'] == 1:
            if self.use_lstm:
                return self.state_encoder.get_net(
                    tf.concat([state[:, 1:, :], system_input[:, 0:-1, :]], axis=len(state.shape) - 1))
            else:
                return self.state_encoder.get_net(state)
        if self.params['concept'] == 2:
            if self.use_lstm:
                lin_state_part = self.state_encoder.get_net(
                    tf.concat([state[:, 1:, :], system_input[:, 0:-1, :]], axis=len(state.shape) - 1))
                return tf.concat([state[:, 1:, :], lin_state_part], axis=len(state.shape) - 1)
            else:
                # todo
                if len(state.shape) == 2:
                    concatState = state[:, :int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))]
                else:
                    concatState = state[:, :, :int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))]
                # todo

                return tf.concat([concatState, self.state_encoder.get_net(state)], axis=len(state.shape) - 1)
        else:
            ValueError('Selected concept does not exist')

    def get_input_encoder(self, state, system_input):
        """
        :param state: state[i,:] =  state_i (state_i is a state of batch element i)
        :param system_input: system_input[i,:] =  system_input_i (system_input_i is a system_input of batch element i)
        :return: tensor of input encoder net with input tensors state and system_input
        """
        if self.params['is_input_affine']:
            # only first input (current input) should be transformed
            return self.input_encoder.get_net(system_input[:, 0:self.params['widths_input_encoder'][0]])
        else:
            return self.input_encoder.get_net(tf.concat([state, system_input], axis=1))

    def get_input_encoder_batch(self, state, system_input):
        """
        :param state: state with 3D Batch state[j, i, :]
        :param system_input:
        :return:
        """
        if self.params['is_input_affine']:
            # only first input (current input) should be transformed
            return self.input_encoder.get_net(system_input[:, :, 0:self.params['widths_input_encoder'][0]])
        else:
            return self.input_encoder.get_net(tf.concat([state, system_input], axis=2))

    def get_state_decoder(self, lin_state):
        """
        same as get_state_encoder() just for decoder
        :param lin_state:
        :return:
        """
        if self.params['concept'] == 1:
            return self.state_decoder.get_net(lin_state)
        if self.params['concept'] == 2:
            # return first elements of linear state vector
            if len(lin_state.shape) == 2:
                return tf.identity(
                    lin_state[:, 0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))])
            elif len(lin_state.shape) == 3:
                return tf.identity(lin_state[:, :,
                                   0:int(self.params['dim_system_state'] / (self.params['delay_coordinates_delay'] + 1))])
            else:
                ValueError('Wrong lin state rank')
        else:
            ValueError('Selected concept does not exist')

    def get_input_decoder(self, state, lin_system_input):
        """
        same as get_input_encoder() just for decoder inputs are 2D tensors
        :param state:
        :param lin_system_input:
        :return:
        """
        if self.params['is_input_affine']:
            return self.input_decoder.get_net(lin_system_input)
        else:
            return self.input_decoder.get_net(tf.concat([state, lin_system_input], axis=1))

    def get_input_decoder_batch(self, state, lin_system_input):
        """
        same as get_input decoder but for 3D vectors
        :param state:
        :param lin_system_input:
        :return:
        """
        if self.params['is_input_affine']:
            return self.input_decoder.get_net(lin_system_input)
        else:
            return self.input_decoder.get_net(tf.concat([state, lin_system_input], axis=2))

    def get_lin_model(self, lin_state, lin_input):
        """
        :param lin_state: state of linear model
        :param lin_input: input of linear model
        :return: tensor of next lin_state
        """
        print(self.K.shape)
        print(lin_state.shape)
        return tf.matmul(lin_state, self.K) + tf.matmul(lin_input, self.L)

    def get_lin_system_matrix(self):
        return self.K

    def get_lin_input_vector(self):
        return self.L

    def get_trainable_vars_state_encoder(self):
        return self.state_encoder.get_trainable_vars()

    def get_trainable_vars_input_encoder(self):
        return self.input_encoder.get_trainable_vars()

    def get_trainable_vars_state_decoder(self):
        return self.state_decoder.get_trainable_vars()

    def get_trainable_vars_input_decoder(self):
        return self.input_decoder.get_trainable_vars()

    def get_weights_state_encoder(self):
        return self.state_encoder.get_weights()

    def get_weights_input_encoder(self):
        return self.input_encoder.get_weights()

    def get_weights_state_decoder(self):
        return self.state_decoder.get_weights()

    def get_weights_input_decoder(self):
        return self.input_decoder.get_weights()

    def get_biases_state_encoder(self):
        return self.state_encoder.get_biases()

    def get_biases_input_encoder(self):
        return self.input_encoder.get_biases()

    def get_biases_state_decoder(self):
        return self.state_decoder.get_biases()

    def get_biases_input_decoder(self):
        return self.input_decoder.get_biases()

    def set_state_encoder(self, state_encoder):
        self.state_encoder = state_encoder

    def set_input_encoder(self, input_encoder):
        self.input_encoder = input_encoder

    def set_state_decoder(self, state_decoder):
        self.state_decoder = state_decoder

    def set_input_decoder(self, input_decoder):
        self.input_decoder = input_decoder

