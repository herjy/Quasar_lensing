import numpy as np
from interpolation_functions import T_K,least_squares, gradient_descent
from simulation_functions import simulation_



sample = np.loadtxt('sample.txt')
t_s = np.sort(sample[0:5476]) # 0:5476 covers all the  array unless sample is edited for the truth to be included at the end of it

T_K_ = T_K(8,sample)  # here you can change the step of desired sampling

f_s = simulation_(sample)


#least_squares(t_s, T_K_,f_s) #uncomment to run sinc interpolation using least_squares

# sinc interpolation using gradient_descent 

gradient_descent(t_s,T_K_,f_s)

