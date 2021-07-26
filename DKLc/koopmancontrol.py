'''
Simple system data generator
'''

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate
import DKLc.networkloader
import tensorflow as tf
import numpy as np
import pickle
import misc.elperfns
import scipy
import scipy.linalg
import scipy.optimize
import qpsolvers
import os


class KoopmanControl:
    def __init__(self, system_ode, delta_t, Q, R, model_path, saving_path, state_norm, max_input, steady_state=0,
                 steady_input=0, input_norm=0, prediction_horizon=5):
        """
        constructor of class KoopmanControl
        :param system_ode: function for system ode
        :param delta_t: sampling time of the controller in seconds
        :param Q: Q-Matrix for LQR
        :param R: R-Matrix for LQR
        :param model_path: path for the model files
        :param saving_path: saving path for the trajecotries
        :param state_norm: norm for the state wich is used in neural networks
        :param max_input: maximum deviation of input around steady_input
        :param steady_state:
        :param steady_input:
        :param input_norm: norm for the input. if zero (default) 1/max_input will be input norm
        """
        self.system_ode = system_ode
        self.delta_t = delta_t
        self.Q = Q
        self.R = R
        self.model_path = model_path
        self.saving_path = saving_path
        self.state_norm = state_norm
        self.max_input = max_input
        if input_norm == 0:
            self.input_norm = np.divide(1, self.max_input)
        else:
            self.input_norm = input_norm
        self.steady_state = steady_state
        self.steady_input = steady_input
        self.params = pickle.load(open(model_path + 'params.p', 'rb'))
        # load network
        self.begin = int(self.params['dim_system_input'] / (self.params['delay_coordinates_delay'] + 1))
        self.state_tensor = tf.placeholder(
            dtype=tf.float32, shape=(None, self.params['dim_system_state']), name='states')
        self.lin_state_tensor = tf.placeholder(
            dtype=tf.float32, shape=(None, self.params['dim_lin_system']), name='states')
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, self.begin), name='inputs')
        self.input_tensor_long = tf.placeholder(
            dtype=tf.float32, shape=(None, self.params['dim_system_input']), name='inputs')

        self.sess = tf.Session()
        self.loader = networkloader.NetworkLoader(model_path, self.sess, self.params, load_tf_graph=False)
        _, end = self.input_tensor_long.shape
        self.state_encoder = self.loader.get_state_encoder(self.state_tensor, self.input_tensor_long)
        self.input_decoder = self.loader.input_decoder.get_net(self.input_tensor)
        self.input_encoder = self.loader.input_encoder.get_net(self.input_tensor)
        self.state_decoder = self.loader.get_state_decoder(self.lin_state_tensor)

        self.K = np.transpose(self.loader.K_numeric)
        self.L = np.transpose(self.loader.L_numeric)

        self.prediction_horizon = prediction_horizon

        self.u_static, = self.sess.run(self.input_encoder,
                                       feed_dict={self.input_tensor: np.zeros((1, self.begin))})
        # generate saving path if necessary
        os.makedirs(saving_path, exist_ok=True)

    def dlqr(self, A, B, Q, R):
        """Solve the discrete time lqr controller.

        x[k+1] = A x[k] + B u[k]

        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """

        # ref Bertsekas, p.151

        # first, try to solve the ricatti equation
        X = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))

        # compute the LQR gain
        bXb = np.matmul(np.matmul(B.T, X), B)
        bXA = np.matmul(np.matmul(B.T, X), A)
        K = np.matmul(np.linalg.inv(bXb + R), bXA)

        return K

    def get_t_95(self, states):
        '''
        calcualted t95 for controll trajectory
        :param states: control trajectory. first dimesnion is the time second is the state
        :return:
        '''
        # calculate relative control deviation
        states_deviation = np.divide((states - self.steady_state), (states[0, :] - self.steady_state))
        i = len(states_deviation) - 1
        while i >= 0:
            for j in range(0, len(states_deviation[i, :])):
                if abs(states_deviation[i, j]) > 0.05:
                    return i
            i -= 1
        return i

    def simulate_system(self, starting_state, save_history=False, horizon=50, Q=None, saving_path=None,
                        control_matrix=None, controller_selection=0):
        """h
        simulates the system
        :param starting_state:
        :param save_history:
        :param horizon:
        :param Q: if None (default) standard Q that was defined in contructor is used
        :param saving_path: if None (default) standard path definden in constructor is used
        :param control_matrix:
        :param controller_selection: 0 for linear controller, 1 for MPC controller
        :return:
        """

        if Q is None:
            Q = self.Q
        if controller_selection == 0:
            if control_matrix is None:
                if Q is not None:
                    control_matrix = self.dlqr(self.K, self.L, Q, self.R)
                else:
                    control_matrix = self.dlqr(self.K, self.L, self.Q, self.R)

        print(control_matrix)
        # init states
        state = starting_state
        state_normalized = np.array([np.multiply(np.ndarray.flatten(state) - self.steady_state, self.state_norm)])

        u_maxs = np.zeros(self.params['dim_system_input'])

        state_history = None
        input_history = None
        lin_state_history = None
        lin_state_history_ref = None
        state_history_ref = None
        lin_state_R, = self.sess.run(self.state_encoder,
                                     feed_dict={self.state_tensor: np.zeros((1, self.params['dim_system_state'])),
                                                self.input_tensor_long: np.zeros((1, self.params['dim_system_input']))})

        #print(lin_state_R)
        # because when u_lin ist zero u_nonlin is not zero, this offset has to be cleaned
        u_static, = self.sess.run(self.input_encoder,
                                  feed_dict={self.input_tensor: np.zeros((1, self.begin))})

        max_lin_inputs = np.ones((self.params['dim_system_input'])) * 0.8

        input_linear_fit_matrix = 0

        # deprecated
        if controller_selection == 1:
            scale = 0.9
            nmbr = 10000
            inputs = (np.random.rand(nmbr, self.params['dim_system_input'])*1.0) * np.random.choice([-1, 1], (nmbr, self.params['dim_system_input']))
            linear_inputs = self.sess.run(self.input_encoder,
                                          feed_dict={self.input_tensor: inputs})
            # fits the nonlinear encoder s.t.
            input_linear_fit_matrix = np.linalg.lstsq(inputs, linear_inputs - u_static)[0].T
            print(inputs.T[:, 1])
            print(input_linear_fit_matrix @ inputs.T[:, 1])
            print(linear_inputs.T[:, 1] - u_static)
            print(np.max(inputs.T[:, :] - np.linalg.inv(input_linear_fit_matrix) @ (linear_inputs[:, :] - u_static).T, axis=1))

        lin_state_ref = 0
        u_limit_exceeded = False
        delay = 0
        if 'delay_coordinates_delay' in self.params:
            delay = self.params['delay_coordinates_delay']
        for i in range(1, horizon):
            u = self.steady_input

            # control law can be applied (controller init is finished)
            if i > delay:
                state_list = state_normalized
                if 'system_outputs' in self.params:
                    state_list = state_normalized[:, self.params['system_outputs']]
                # first (current) input os not important but has to be inputted into state_enocder
                input_list = np.zeros((1, self.begin))
                if delay != 0:
                    for j in range(1, delay+1):
                        end, _ = state_history.shape
                        temp = np.array([np.multiply(state_history[end - j, self.params['system_outputs']]
                                                     - self.steady_state[self.params['system_outputs']],
                                                     self.state_norm[self.params['system_outputs']])])
                        state_list = np.concatenate((state_list, temp), axis=1)
                        print(temp)
                    for j in range(0, delay):
                        _, end = input_history.shape
                        temp = np.array([np.multiply(input_history[end - j, :] - self.steady_input, self.input_norm)])
                        input_list = np.concatenate((input_list, temp), axis=1)

                # calculate control deviation
                #print(self.state_tensor.shape)
                #print(state_list)
                #print(state_history)
                lin_state, = self.sess.run(self.state_encoder, feed_dict={self.state_tensor: state_list,
                                                                          self.input_tensor_long: input_list})

                # calcuate neural network prediction
                if lin_state_history_ref is None:
                    lin_state_ref = np.array([lin_state]).T
                state_ref, = self.sess.run(self.state_decoder, feed_dict={self.lin_state_tensor: lin_state_ref.T})
                state_ref = np.divide(state_ref, self.state_norm) + self.steady_state

                controller_output_lin = np.zeros((self.params['dim_system_input'], 1))
                # calculate control law
                if controller_selection == 0:
                    controller_output_lin = np.matmul(control_matrix, np.transpose([lin_state_R - lin_state]))
                    controller_output_lin = controller_output_lin.T + np.array([u_static])
                elif controller_selection == 1:
                    # apply mpc
                    mpc_input_traj = self.model_predictive_control(
                        np.array([lin_state - lin_state_R]).T, Q, max_lin_inputs, input_linear_fit_matrix)
                    print(np.linalg.inv(input_linear_fit_matrix) @ mpc_input_traj[0:self.params['dim_system_input'], :])
                    controller_output_lin = mpc_input_traj[0:self.params['dim_system_input'], :].T + np.array([u_static])
                elif controller_selection == 2:
                    # apply mpc
                    mpc_input_traj = self.model_predictive_control_nonlinear(
                        np.array([lin_state - lin_state_R]).T, Q, max_lin_inputs)
                    controller_output_lin = mpc_input_traj[0:self.params['dim_system_input'], :].T + np.array([u_static])

                # decoder lin_input
                u, = self.sess.run(self.input_decoder, feed_dict={self.input_tensor: controller_output_lin})
                #if i < 20:
                #    u = np.array([-0.2, -0.2, 0.6])
                #else:
                #    u = np.array([0.0, 0.0, 0.0])
                print(u)
                print('########################')

                # calculate NN ref
                v, = self.sess.run(self.input_encoder, feed_dict={self.input_tensor: np.array([u])})
                lin_state_ref = np.matmul(self.K, lin_state_ref) + np.matmul(self.L, np.array([v]).T)

                # denormalize input
                u = np.divide(u, self.input_norm) + self.steady_input
                print(u)

                # check if limit is exceeded (important for evo strat)
                if np.any(u < self.steady_input - self.max_input*0.9) or np.any(u > self.steady_input + self.max_input*0.9):
                    u_limit_exceeded = True
                if i == delay + 1:
                    u_maxs = np.abs(u) - self.steady_input
                else:
                    for j in range(0, len(u_maxs)):
                        if u_maxs[j] < abs(u[j] - self.steady_input[j]):
                            u_maxs[j] = abs(u[j] - self.steady_input[j])

                # saturate input
                u = np.clip(u, self.steady_input - self.max_input, self.steady_input + self.max_input)

                # save nn trajectory
                if lin_state_history_ref is None:
                    lin_state_history = np.array([lin_state])
                    lin_state_history_ref = np.transpose(lin_state_ref)
                    state_history_ref = np.array([state_ref])
                else:
                    lin_state_history = np.concatenate((lin_state_history, np.array([lin_state])), axis=0)
                    lin_state_history_ref = np.concatenate((lin_state_history_ref, np.transpose(lin_state_ref)), axis=0)
                    state_history_ref = np.concatenate((state_history_ref, np.array([state_ref])), axis=0)

            # save trajectory
            if state_history is None:
                state_history = np.array([state])
                input_history = np.array([u])
            else:
                state_history = np.concatenate((state_history, np.array([state])), axis=0)
                input_history = np.concatenate((input_history, np.array([u])), axis=0)

            # simulate system
            states_temp = odeint(self.system_ode, np.ndarray.flatten(state), np.array([0, self.delta_t]),
                                 args=(u,), rtol=1e-12)
            state_normalized = np.array([np.multiply(states_temp[1, :] - self.steady_state, self.state_norm)])
            state = states_temp[1, :]

        if save_history:
            if saving_path is None:
                saving_path = self.saving_path
            # generate saving path if necessary
            os.makedirs(saving_path, exist_ok=True)
            np.savetxt(saving_path + 'states.csv', state_history)
            np.savetxt(saving_path + 'states_ref.csv', state_history_ref)
            np.savetxt(saving_path + 'lin_states.csv', lin_state_history)
            np.savetxt(saving_path + 'inputs.csv', input_history)
            np.savetxt(saving_path + 'lin_states_ref.csv', lin_state_history_ref)

        # calculate retrun values like T95 and max deviation
        u_maxs_dev = 0
        if u_limit_exceeded:
            u_maxs = u_maxs - self.max_input * 0.9
            u_maxs = np.clip(u_maxs, 0, self.max_input*100)
            u_maxs_dev = u_maxs_dev + np.linalg.norm(u_maxs * u_maxs)
        t_95 = self.get_t_95(state_history) / horizon
        dev = np.divide(state_history-self.steady_state, state_history[0, :]-self.steady_state)
        max_dev = abs(dev.min()) + 0.3 * dev.max() # + abs(dev.min())
        return t_95, max_dev, u_maxs_dev

    def model_predictive_control(self, lin_state, state_weight, max_lin_input, input_trans_matrix=None):
        """
        linear mpc controller (controls state to 0)
        :param lin_state: linear starting state
        :param state_weight: weight for states (Q-Matrix in doc)
        :param max_lin_input: maximum linear input
        :param input_trans_matrix: matrix M that transforms input u s.t u=M*v if not none input boundaries are
        threated as v<max_lin_input
        :return: control inputs put into an
        """
        if input_trans_matrix is None:
            input_trans_matrix = np.eye(self.params['dim_system_input'])
        # invert matrix
        # repeat trans matrix at diagonal.
        b = [input_trans_matrix] * self.prediction_horizon
        input_trans_matrix = scipy.linalg.block_diag(*b)
        # generate matrizes for linear mpc
        transition_matrix, control_matrix = self.generate_control_matrix()
        control_matrix = control_matrix @ input_trans_matrix
        b = [state_weight] * self.prediction_horizon
        Q = scipy.linalg.block_diag(*b)
        q = np.matmul(np.matmul(np.matmul(control_matrix.T, Q), transition_matrix), lin_state).reshape(
            (self.params['dim_system_input']*self.prediction_horizon))
        P = np.matmul(np.matmul(control_matrix.T, Q), control_matrix)
        P = P + np.eye(len(P)) * 0.000001

        # repeat maximum input
        max_lin_input = np.tile(max_lin_input, self.prediction_horizon)

        # convert matrizes to double
        P = np.array(P, dtype=np.double)
        q = np.array(q, dtype=np.double)
        max_lin_input = np.array(max_lin_input, dtype=np.double)

        # solve qp problem
        input_traj = qpsolvers.solve_qp(P, q, lb=-max_lin_input, ub=max_lin_input)

        input_traj = np.array(input_traj, dtype=np.float32)
        print(input_traj)
        input_traj = np.array([input_traj], dtype=np.float32).T
        return input_trans_matrix @ input_traj

    def model_predictive_control_nonlinear(self, lin_state, state_weight, max_lin_input):
        """
        linear mpc controller (controls state to 0)
        :param lin_state: linear starting state
        :param state_weight: weight for states (Q-Matrix in doc)
        :param max_lin_input: maximum linear input
        :param input_trans_matrix: matrix M that transforms input u s.t u=M*v if not none input boundaries are
        threated as v<max_lin_input
        :return: control inputs put into an
        """
        # generate matrizes for linear mpc
        transition_matrix, control_matrix = self.generate_control_matrix()
        b = [state_weight] * self.prediction_horizon
        Q = scipy.linalg.block_diag(*b)
        q = np.matmul(np.matmul(np.matmul(control_matrix.T, Q), transition_matrix), lin_state).reshape(
            (self.params['dim_system_input'] * self.prediction_horizon))
        P = np.matmul(np.matmul(control_matrix.T, Q), control_matrix)

        # repeat maximum input
        max_lin_input = np.tile(max_lin_input, self.prediction_horizon)

        # convert matrizes to double
        '''P = np.array(P, dtype=np.double)
        q = np.array(q, dtype=np.double)
        max_lin_input = np.array(max_lin_input, dtype=np.double)'''

        constr = scipy.optimize.NonlinearConstraint(self.mpc_constraints, -max_lin_input, max_lin_input)
        constr = {'type': 'ineq', 'fun': self.mpc_constraints}
        # solve qp problem
        input_traj_sol = scipy.optimize.minimize(self.mpc_objective,
                                                 x0=np.zeros((self.params['dim_system_input'] * self.prediction_horizon,)),
                                                 args=(P, q), constraints=constr, options={'maxiter': 5000},
                                                 method='COBYLA')

        print(input_traj_sol.message)

        input_traj = np.array(input_traj_sol.x, dtype=np.float32)
        print(input_traj)
        input_traj = np.array([input_traj], dtype=np.float32).T
        print('Hello There')
        print(self.sess.run(self.input_decoder,
                               feed_dict={self.input_tensor:
                                          np.reshape(input_traj.T,
                                                     (self.prediction_horizon, self.params['dim_system_input'])) + self.u_static.T}))
        return input_traj

    def mpc_objective(self, inputs, P, q):
        return 0.5 * np.array([inputs]) @ P @ np.array([inputs]).T + q.T @ np.array([inputs]).T

    def mpc_constraints(self, inputs):
        inputs = self.sess.run(self.input_decoder,
                               feed_dict={self.input_tensor:
                                          np.reshape(inputs,
                                                     (self.prediction_horizon, self.params['dim_system_input'])) + self.u_static.T})
        if np.any(np.reshape(inputs, (self.prediction_horizon * self.params['dim_system_input'],)) > 1.0):
            return -1
        if np.any(np.reshape(inputs, (self.prediction_horizon * self.params['dim_system_input'],)) < -1.0):
            return -1
        return 1

    def generate_control_matrix(self):
        """
        generates transition and control matrix for mpc controller
        :return: transition_matrix, control_matrix
        """
        _, p = self.L.shape
        n, _ = self.K.shape
        control_matrix = np.zeros((self.prediction_horizon * n, self.prediction_horizon * p))
        # put b into triangle
        for row in range(0, self.prediction_horizon):
            for column in range(0, row + 1):
                control_matrix[n * row:n * (row + 1), p * column:p * (column + 1)] = self.L

        # multiply by A
        for k in range(0, self.prediction_horizon):
            for row in range(k + 1, self.prediction_horizon):
                for column in range(0, row - k):
                    control_matrix[n * row:n * (row + 1), p * column:p * (column + 1)] = \
                        np.matmul(self.K, control_matrix[n * row:n * (row + 1), p * column:p * (column + 1)])

        transition_matrix = np.zeros((self.prediction_horizon * n, n))
        for row in range(0, self.prediction_horizon):
            transition_matrix[n * row:n * (row + 1), :] = self.K
        for k in range(1, self.prediction_horizon):
            for row in range(k, self.prediction_horizon):
                transition_matrix[n * row:n * (row + 1), :] = \
                    np.matmul(self.K, transition_matrix[n * row:n * (row + 1), :])

        return transition_matrix, control_matrix

    def evolutionary_strategy(self, starting_state):
        mu = 10
        lmbda = 10 * 10
        t0 = 0.1
        t1 = 0.1
        k_t95 = 5
        k_dev = 5
        individuals = np.abs(np.random.randn(len(self.K), mu) * 1)
        sigmas = np.abs(np.random.randn(len(self.K), mu))
        sigmas_max = 2
        sigmas_min = 0.001
        steps = 300
        for i in range(0, steps):
            # clone
            clones_individuals = np.repeat(individuals, axis=1, repeats=int(lmbda/mu))
            clones_sigmas = np.repeat(sigmas, axis=1, repeats=int(lmbda/mu))

            # mutate
            clones_sigmas = clones_sigmas * np.exp(
                0.1*np.random.randn(lmbda)) * np.exp(0.1*np.random.randn(len(self.K), lmbda))
            clones_sigmas = np.clip(clones_sigmas, sigmas_min, sigmas_max)
            clones_individuals = clones_individuals + clones_sigmas * np.random.randn(len(self.K), lmbda)
            clones_individuals = np.clip(clones_individuals, 0, 1000)
            fitness = np.zeros((lmbda,))
            t95s = np.zeros((lmbda,))
            max_devs = np.zeros((lmbda,))
            for j in range(0, lmbda):
                Q = np.diag(clones_individuals[:, j])
                t95, max_dev, u_maxs_dev = self.simulate_system(starting_state, Q=Q)
                fitness[j] = k_t95 * t95 + k_dev * max_dev + u_maxs_dev * 10
                #print(u_maxs_dev)
                t95s[j] = t95
                max_devs[j] = max_dev
            ind_sort = np.argsort(fitness)
            individuals = clones_individuals[:, ind_sort[0:mu]]
            diag = ''
            for ind in range(0, len(self.K)):
                diag = diag + str(individuals[ind, 0]) + ', '
            print('fitness best: ' + str(fitness[ind_sort[mu-1]]))
            print(t95s[ind_sort[mu-1]])
            print(max_devs[ind_sort[mu-1]])
            print(diag)
