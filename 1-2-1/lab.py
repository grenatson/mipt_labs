import numpy as np
from my_stat import LeastSquares, np_to_latex


M = 2925 / 1000 #+-5
g = 9.81
L = 217.6 / 100
bullets = np.array([0.515, 0.507, 0.506, 0.500, 0.496, 0.509, 0.505, 0.508])
'''
np.set_printoptions(precision=2)

b_first = bullets[4:]
x_1 = np.array([7.75, 7.5, 7.0, 4.0])
x0 = np.array([-2, -2.25, -2.75, -3, -3.5])
x_before = x_1 - x0[:-1]
x_after = x_1 - x0[1:]

v_before = M / (b_first / 1000) * (g / L)**0.5 * x_before / 1000
v_after = M / (b_first / 1000) * (g / L)**0.5 * x_after / 1000

v_spec = np.array([124.1, 126.2, 113.1, 94.87])

#np_to_latex([b_first, x_before, x_after, v_before, v_after, v_spec])

u, sigma = np.mean(v_after), np.std(v_after, ddof=1)
syst = u * (1 / 500**2 + (5/3000)**2 + 0.25 * (0.5/217.6)**2 + 1 / 25)**0.5
deviation = (syst**2 + (sigma/2)**2)**0.5
print(u, sigma, syst, deviation)
print(np.mean(v_spec), np.std(v_spec, ddof=1))
'''

#=second part=#
M = (724.5 + 725.8) / 2 / 1000 #+-0.1
r = 20.5 / 100
R = 34 / 100
d = 56 / 100
T1 = 17.34
T2 = 12.79

b_second = bullets[:4] / 1000
x = np.array([13.7, 15.0, 13.0, 12.0]) / 100
x_0 = np.array([-2.25, -2.25, -2.5, -2]) / 100

u = 2 * np.pi * (x - x_0) * M / d / b_second * R*R / r * T1 / (T1*T1 - T2*T2)
print(u)

v, sigma = np.mean(u), np.std(u, ddof=1)
syst = v * ((0.5 / 13.7)**2 + (1/500)**2 + (0.1/725)**2 + (0.5/56)**2 + (0.1/20.5)**2 + 4 * (0.1/34)**2 + (0.3/12/17.34)**2 + (2 * (0.3/12)**2 /(17.34**2 - 12.79**2)))**0.5
deviation = (syst**2 + (sigma/2)**2)**0.5
print(sigma, syst)
print(v, deviation)