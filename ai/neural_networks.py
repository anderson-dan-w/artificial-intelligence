#!/usr/bin/env python3
import random
import math

def convert_csv_to_data(fname):
    fh = open(fname)
    data = fh.read().replace("\r","\n").replace("\n\n", "\n").split("\n")
    fh.close()
    data_list = [[float(v) for v in d.replace("\n","").split(",")] for d in data]
    return data_list

#fname =  "pr7_auto_mpg_small.csv"
#data_list = convert_csv_to_data(fname)

class NeuralNetwork(object):
    def __init__(self, data_list, hidden_nodes=None, output_nodes=1, alpha=0.1):
        self.n_output_nodes = int(output_nodes)
        self.n_input_nodes = len(data_list[0]) - self.n_output_nodes + 1
        self.n_hidden_nodes = hidden_nodes
        if self.n_hidden_nodes is None:
            self.n_hidden_nodes = self.n_input_nodes
        
        self.output_vectors = [d[:self.n_output_nodes] for d in data_list]
        self.output_weights = [[random.random() for j in 
            range(self.n_hidden_nodes + 1)] for i in range(self.n_output_nodes)]
        self.output_additives = [[0 for j in range(self.n_hidden_nodes 
            + 1)] for i in range(self.n_output_nodes)]
        self.output_zs = [0] * self.n_output_nodes
        self.output_y_hats = [0] * self.n_output_nodes
        self.delta_output_errors = [0] * self.n_output_nodes
        self.errors = [0] * self.n_output_nodes
        
        self.input_vectors = [[1] + d[self.n_output_nodes:] for d in data_list]
        
        self.hidden_weights = [[random.random() for j in 
            range(self.n_input_nodes)] for i in range(self.n_hidden_nodes)]
        self.hidden_additives = [[0 for j in range(self.n_input_nodes)]
                for i in range(self.n_hidden_nodes)]
        self.hidden_zs = [0] * self.n_hidden_nodes
        self.hidden_y_hats = [1] + [0] * (self.n_hidden_nodes)
        self.delta_hidden_errors = [0] * (self.n_hidden_nodes + 1)
        
        self.alpha = alpha

    def calc_hidden_zs(self, vec):
        """ where z is the activity function; linear accumlator"""
        for i in range(self.n_hidden_nodes):
            s = sum(vec[j] * self.hidden_weights[i][j]
                    for j in range(self.n_input_nodes))
            self.hidden_zs[i] = s
        return

    def calc_hidden_y_hats(self):
        """ where y_hat is the activation function; sigmoid"""
        for i in range(self.n_hidden_nodes):
            y_hat = 1.0/(1 + math.e**min(-self.hidden_zs[i], 100))
            self.hidden_y_hats[i+1] = y_hat
        return

    def calc_output_zs(self):
        """ z = activity function output; linear accumulator
            +1 hidden node, for bias
        """
        for i in range(self.n_output_nodes):
            s = sum(self.hidden_y_hats[j]*self.output_weights[i][j]
                    for j in range(self.n_hidden_nodes + 1))
            self.output_zs[i] = s
        return

    def calc_output_y_hats(self):
        """ y_hat is activation function output; sigmoid"""
        for i in range(self.n_output_nodes):
            y_hat = 1.0/(1 + math.e**min(-self.output_zs[i], 100))
            self.output_y_hats[i] = y_hat
        return

    def calc_error(self, output_vec):
        for i in range(self.n_output_nodes):
            self.errors[i] = output_vec[i] - self.output_y_hats[i]
        return

    def calc_squared_error(self, debug=False):
        squared_error = 0
        for in_vec, out_vec in zip(self.input_vectors, self.output_vectors):
            self.feed_forward(in_vec)
            self.calc_error(out_vec)
            sq = sum(e**2 for e in self.errors)
            if debug:
                print("sq_error for {} is {}".format(in_vec, sq))
            squared_error += sq
        return squared_error/2

    def calc_delta_output_errors(self):
        """ Calculate error from output vector"""
        y_hats = self.output_y_hats
        for i in range(self.n_output_nodes):
            error = y_hats[i] * (1 - y_hats[i]) * self.errors[i]
            self.delta_output_errors[i] = error
        return

    def calc_output_weight_additives(self):
        """ j+1 to skip over output bias"""
        for i in range(self.n_output_nodes):
            for j in range(self.n_hidden_nodes):
                additive = (self.alpha * self.delta_output_errors[i] *
                            self.hidden_y_hats[j+1])
                self.output_additives[i][j+1] += additive
        return

    def calc_delta_hidden_errors(self):
        for i in range(self.n_hidden_nodes + 1):
            s = sum(self.delta_output_errors[j] * self.output_weights[j][i]
                    for j in range(self.n_output_nodes))
            err = (self.hidden_y_hats[i] * (1 - self.hidden_y_hats[i]) * s)
            self.delta_hidden_errors[i] = err
        return

    def calc_hidden_weight_additives(self, vec):
        """ i+1 skips over the bias for hidden nodes;
            j = range(1, n) skips over the bias for the input nodes
        """
        for i in range(self.n_hidden_nodes):
            for j in range(1, self.n_input_nodes):
                additive = self.alpha * self.delta_hidden_errors[i+1] * vec[j]
                self.hidden_additives[i][j] += additive
        return

    def update_weights(self, mode="online"):
        """ order agnostic, since it's a 'simultaneous' update of all the weight
            additives derived so far (each additive[i][j] will be a sum over
            multiple input vectors' additives, if in batch mode)
        """
        if mode == "online":
            divisor = 1
        else:
            divisor = len(self.output_vectors)
        for i in range(self.n_hidden_nodes):
            for j in range(self.n_input_nodes):
                avg_additive = self.hidden_additives[i][j] / divisor
                self.hidden_weights[i][j] += avg_additive
                self.hidden_additives[i][j] = 0 ## clear it out for next iter
        for i in range(self.n_output_nodes):
            for j in range(self.n_hidden_nodes + 1):
                avg_additive = self.output_additives[i][j] / divisor
                self.output_weights[i][j] += avg_additive
                self.output_additives[i][j] = 0

    def feed_forward(self, input_vector):
        self.calc_hidden_zs(input_vector)
        self.calc_hidden_y_hats()
        self.calc_output_zs()
        self.calc_output_y_hats()
        return          
    
    def back_propogation(self, input_vector, output_vector):
        self.calc_error(output_vector)
        self.calc_delta_output_errors()
        self.calc_output_weight_additives()
        self.calc_delta_hidden_errors()
        self.calc_hidden_weight_additives(input_vector)
        return

    def train_one(self, input_vector, output_vector):
        self.feed_forward(input_vector)
        self.back_propogation(input_vector, output_vector)

    def online_training(self, epsilon=.0000001, max_iters=15, debug=False):
        """ Online training calculates the output and hidden additives for each
            vector and then updates the weights prior to the next input, output
            pair being run
        """
        for i in range(max_iters):
            for in_vec, out_vec in zip(self.input_vectors, self.output_vectors):
                self.train_one(in_vec, out_vec)
                self.update_weights()
            squared_error = self.calc_squared_error()
            if debug:
                print("After {}, squared_error = {}".format(i, squared_error))
            if squared_error < epsilon:
                print("Went below epsilon threshold ({})".format(epsilon))
                return
        return

    def batch_training(self, epsilon=.0000001, max_iters=15, debug=False):
        """ Batch training calculates all output and hidden additives in a
            cumulative fashion, and doesn't update the weights until after every
            input and out vector has been examined
        """
        for i in range(max_iters):
            for in_vec, out_vec in zip(self.input_vectors, self.output_vectors):
                self.train_one(in_vec, out_vec)
            self.update_weights(mode="batch")
            squared_error = self.calc_squared_error()
            if debug:
                print("After {}, squarederror = {}".format(i, squared_error))
            if squared_error < epsilon:
                print("Went below epsilon threshold ({})".format(epsilon))
                return
        return

    def predict(self, input_vector, binary=False):
        input_vector = input_vector
        self.feed_forward(input_vector)
        if binary:
            return [round(y_hat) for y_hat in self.output_y_hats]
        return self.output_y_hats

## for testing
if False:
    data = [[1, 2.7, 3.9]] # output, followed by 2 input nodes
    hidden_weights = [[0.3, 0.9, -0.4], [0.3, 0.4, -0.9]]
    output_weights = [[0.5, 0.6, -0.3]]

    n = NeuralNetwork(data, hidden_nodes=2)
    n.hidden_weights = hidden_weights
    n.output_weights = output_weights
    print("in={n.input_vectors}; hidden_w={n.hidden_weights}; out_w={n.output_weights}; "
          "out_v={n.output_vectors}".format(n=n))

    o1 = n.predict([4.5, 1.7])
    print("o1={}".format(o1))

    n.calc_hidden_zs([1, 2.7, 3.9])
    print("\nhidden_zs={n.hidden_zs}".format(n=n))

    n.calc_hidden_y_hats()
    print("\nhidden_y_hats={n.hidden_y_hats}".format(n=n))

    n.calc_output_zs()
    print("\noutput_zs={n.output_zs}".format(n=n))

    n.calc_output_y_hats()
    print("\noutput_y_hats={n.output_y_hats}".format(n=n))

    n.calc_error([1])

    n.calc_delta_output_errors()
    print("\ndelta_out_err={n.delta_output_errors}".format(n=n))

    n.calc_output_weight_additives()
    print("\noutout_weights={n.output_weights}".format(n=n))

    n.calc_delta_hidden_errors()
    print("\ndelta_hidden_errors={n.delta_hidden_errors}".format(n=n))

    n.calc_hidden_weight_additives([1, 2.7, 3.9])
    print("\nnew_hid_weights={n.hidden_weights}".format(n=n))

    n.update_weights()

    fname =  "pr7_auto_mpg_small.csv"
    data_list = convert_csv_to_data(fname)
    n = NeuralNetwork(data_list)
    n.online_training()
    print("output_weights={n.output_weights}".format(n=n))
