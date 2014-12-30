#!/usr/bin/python3

import math

def f(xvec, wvec):
    """ Sigmoid activations function: multiply each x_i*w_i, sum them, then
        plug that in to the typical sigmoid function."""
    s = sum(x*w for x,w in zip(xvec, wvec))
    return 1/(1+math.e**(-1 * s))

def delta(xvec, wvec, desired):
    """ Calculate delta_j = -e_j * (1 - y_j) * (y_j) """
    yj = f(xvec, wvec)
    return (desired - yj) * (1 - yj) * (yj)

def delta_weights(xvec, wvec, desired, eta):
    """ Return a vector of weight-deltas. This is how much each weight should
        change by, not the new weight itself. Take care to have a delta=0 for
        the bias, so that it doesn't change"""
    delta_wvec = []
    d = delta(xvec, wvec, desired)
    for x in xvec[:-1]: ## don't want to update the bias, ie last weight
        delta_w = eta * d * x
        delta_wvec.append(delta_w)
    ## delta of the bias is, in effect, zero, i.e. doesn't change
    delta_wvec.append(0)
    return delta_wvec

def new_weights(wvec, delta_wvec):
    """ Return the updated weights. The benefit of separating this from the
        delta weights function is that it becomes easier to calculate multiple
        outputs without accidentally using some updated weights in the middle
        of the calculations (got burned by this in AI)."""
    return [w + dw for w, dw in zip(wvec, delta_wvec)]

def mse(xvec, wvec, desired):
    """ Convenience function for examining mean-squared error"""
    return 0.5*(desired - f(xvec, wvec))**2

def train_nn(xvec, wvec, desired, eta, epsilon, max_iters=100):
    for itr in range(max_iters):
        err = mse(xvec, wvec, desired)
        if err < epsilon:
            print("Below epsilon on itr {}".format(itr))
            break
        wvec = new_weights(wvec, delta_weights(xvec, wvec, desired, eta))
    return wvec

