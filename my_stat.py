import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def variance(array):
    return np.mean((array - np.mean(array)) ** 2)

def standart_deviation(array):
    return np.sqrt(np.mean((np.mean(array) - array) ** 2))

def corrected_standart_deviation(array):
    return np.sqrt(np.sum((np.mean(array) - array) ** 2) / (len(array) - 1))

def covariance(array_1, array_2):
    if len(array_1) != len(array_2):
        raise Exception('Arrays must have the same lenght!')
    return np.mean(array_1 * array_2) - np.mean(array_1) * np.mean(array_2)

class LeastSquares:

    xs = np.array([])
    ys = np.array([])

    def __init__(self, x_data, y_data, pass_through_zero = False):
        if len(x_data) != len(y_data):
            raise Exception('Arrays must have the same lenght!')
        self.xs = x_data
        self.ys = y_data
        self.zero_pass = pass_through_zero
        if self.zero_pass:
            self.k = np.mean(self.xs * self.ys) / np.mean(self.xs ** 2)
            self.sigma_k = np.sqrt((np.mean(self.ys ** 2) / np.mean(self.xs ** 2) - self.k ** 2) / (len(self.xs) - 1))
            self.b = 0
            self.sigma_b = 0
        else:
            self.k = covariance(self.xs, self.ys) / covariance(self.xs, self.xs)
            self.sigma_k = np.sqrt((variance(self.ys) / variance(self.xs) - self.k ** 2) / (len(self.xs) - 2))
            self.b = np.mean(self.ys) - self.k * np.mean(self.xs)
            self.sigma_b = self.sigma_k * np.sqrt(np.mean(self.xs ** 2))
    
    def __repr__(self):
        if self.zero_pass:
            return "y = kx: k = {} \u00b1 {}".format(self.k, self.sigma_k)
        return "y = kx + b: k = {} \u00b1 {}, b = {} \u00b1 {}".format(self.k, self.sigma_k, self.b, self.sigma_b)

    def add_to_axes(self, ax, start = None, end = None, param_dict = {'linewidth': 3}):
        if start == None:
            start = np.min(self.xs)
        if end == None:
            end = np.max(self.xs)
        out = ax.plot(np.array([start, end]), np.array([self.k * start + self.b, self.k * end + self.b]), **param_dict)
        return out
