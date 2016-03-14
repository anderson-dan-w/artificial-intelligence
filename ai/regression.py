import random
import math

##pr7
from ai import ai_utility

def calc_linear_yhats(thetas, x_vectors):
    nthetas = len(thetas)
    return [sum(thetas[i] * vec[i] for i in range(nthetas))
            for vec in x_vectors]
    
def calc_linear_error(y_hats, ys):
    n = len(ys)
    return sum(((y_hats[i] - ys[i])**2) for i in range(n))/(n*2.0)

def calc_logistic_yhats(thetas, x_vectors):
    nthetas = len(thetas)
    zs = [sum(thetas[i] * vec[i] for i in range(nthetas))
          for vec in x_vectors]
    return [1.0/(1 + math.e**(min(-zs[i], 100))) for i in range(len(x_vectors))]

const = 1*10**-200 # prevents ValueErrors in log function.
def calc_logistic_error(y_hats, ys):
    n = len(ys)
    return sum(ys[i] * math.log(y_hats[i] + const) +
                (1 - ys[i]) * math.log(1 - y_hats[i] + const))/(-1 * (n + 0.0))

class Regression(object):
    def __init__(self, data_list, isLinear=True, alpha=.01, epsilon=.000001):
        self.input_vectors = [[1] + d[1:] for d in data_list]
        self.outputs = [d[0] for d in data_list]
        self.n = len(self.input_vectors)
        self.nthetas = len(self.input_vectors[0])
        self.thetas = self.initialize_thetas()
        self.alpha = alpha
        self.epsilon = epsilon
        self.error = 0
        self.theta_derivs = [0] * self.nthetas
        if isLinear:
            self.calc_yhats = calc_linear_yhats
            self.calc_error = calc_linear_error
        else:
            self.calc_yhats = calc_logistic_yhats
            self.calc_error = calc_logistic_error
        
    def initialize_thetas(self):
        self.thetas = [random.random() for i in range(self.nthetas)]
        return
        
    def calc_theta_derivs(self):
        y_hats, ys, inputs = self.y_hats, self.outputs, self.input_vectors
        diffs = [y_hats[i] - ys[i] for i in range(self.n)]
        self.theta_derivs = [sum(((y_hats[i] - ys[i]) * inputs[i][t])
                                   for i in range(self.n))/(self.n+0.0)
                             for t in range(self.nthetas)]
        return
        
    def update_thetas(self):
        t, d = self.thetas, self.theta_derivs
        self.thetas = [t[i] - self.alpha * d[i] for i in range(self.nthetas)]
        
    def converge(self, iters=20000):
        delta_error = self.epsilon + 1 # just so it doesn't break before entering while loop
        itr, maxIters = 0, iters
        while delta_error > self.epsilon and itr < maxIters:
            prev_error = self.error
            self.y_hats = self.calc_yhats(self.thetas, self.input_vectors)
            self.error = self.calc_error(self.y_hats, self.outputs)
            self.calc_theta_derivs()
            self.update_thetas()
            delta_error = prev_error - self.error
            itr += 1
            #print("Now: itr = {i}, err={r.error}  thetas={r.thetas}".format(i=itr, r=self))
        return

def make_pretty_theta_string(thetas):
    return "[" + ", ".join("{:.03}".format(t) for t in thetas) + "]"

linear = "pr7_concrete_compressive_strength_small.csv"
lin_data = ai_utility.convert_csv_to_data(linear)
lr = Regression(lin_data, isLinear=True)
for i in range(10):
    lr.thetas = [random.random()*2 - 1 for j in range(lr.nthetas)]
    start_theta_str = make_pretty_theta_string(lr.thetas)
    lr.converge()
    new_theta_str = make_pretty_theta_string(lr.thetas)
    print("Started with {t};\n\tnow {n}, err = {l.error}\n\n"
            .format(t=start_theta_str, n=new_theta_str, l=lr))

logistic = "pr7_auto_mpg_small.csv"
log_data = convert_csv_to_data(logistic)
lr = Regression(log_data, isLinear=False)
for i in range(10):
    lr.thetas = [random.random()*2 - 1 for j in range(lr.nthetas)]
    start_theta_str = make_pretty_theta_string(lr.thetas)
    lr.converge()
    new_theta_str = make_pretty_theta_string(lr.thetas)
    print("Started with {t};\n\tnow {n}, err = {l.error}\n\n"
            .format(t=start_theta_str, n=new_theta_str, l=lr))



