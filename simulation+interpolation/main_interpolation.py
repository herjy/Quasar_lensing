import numpy as np
import pickle 
import matplotlib.pyplot as plt
from sinc_functions import T_K,least_squares, gradient_descent
from numpy.linalg import inv
from simulation import simulation_



sample = np.loadtxt('sample.txt')
t_s = np.sort(sample[0:5476]) # 0:5476 covers all the  array unless sample is edited for the truth to be included at the end of it

T_K_ = T_K(8,sample)  # here you can change the step of desired sampling

f_s = simulation_(sample)


#least_squares(t_s, T_K_,f_s)

gradient_descent(t_s,T_K_,f_s)

