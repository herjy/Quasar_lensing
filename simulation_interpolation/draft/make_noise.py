from simulation_functions import index, simulation_, plot
from interpolation_functions import T_K
import numpy as np
import matplotlib.pyplot as plt

def size_estimate(i): 
    '''
    inferring the size of the error relative to the magnitude in the real data
    :param i: columns of magnitudes, for this dataset, it is either 1,3,5,7
    :param d: dataset, ie: data
    :return: mean of the fraction sqrt(f1^2 + f2^3+ ...)/mean(error in mag)
    '''
    
    d = np.loadtxt('../RXJ1131_ALL.rdb',skiprows=2)

    f = d[:,i]
    F_error = np.mean(d[:,i+1])
    f2 = []
    for i in range(len(f)):
        f2.append(f[i]**2)
    return np.int((np.sqrt(np.sum(np.array(f2))))/F_error)

def simulate_noise(dt_max, noise=True):
    '''
    adds guassian noise to simulated data
    input:
    dt_max : time delays will be taken randomly between 0 and delay_max
    noise : to not add any noise, set to False, default : True. 
    output:
    ts : simulation time sampling
    fs+noise : noisy magnitude corresponding to ts 
    time_delays: time delay in each of the four curves
    noise_std: the standard deviation of the distribution of noise added to each curve
    '''

    t_drive_new = np.loadtxt('t_drive_new.txt') # true time sample.

    ind = index(dt_max,t_drive_new) 
    # getting the indices of each time shifter curve
    index0, index1,index2,index3 = ind[0],ind[1],ind[2],ind[3]
    sample = np.sort(ind[5])

    #getting the time sample of each curve
    ts = np.array([sample[index0],sample[index1],sample[index2],sample[index3]])

    # finding the magnitudes corresponding to each time shifted sample
    f = simulation_(sample) #running simulation
    fs = np.array([f[index0],f[index1],f[index2],f[index3]])

    time_delays = ind[6]
  

    a,b,c,d = size_estimate(1),size_estimate(3),size_estimate(5),size_estimate(7)

    # calculating the a th, b th, c th, d th of the power of the signal
    def std_gausian(fraction,mag):
        f2 = []
        for i in range(len(mag)):
            f2.append(mag[i]**2)
        return (np.sqrt(np.sum(np.array(f2))))/fraction
    error0,error1,error2,error3 = std_gausian(a,fs),std_gausian(b,fs),std_gausian(c,fs),std_gausian(d,fs)
    noise_std = np.array([error0,error1,error2,error3])

    # generating noise
    noise0= np.random.normal(0,error0,size = len(fs[0]))
    noise1= np.random.normal(0,error1,size = len(fs[1]))
    noise2= np.random.normal(0,error2,size = len(fs[2]))
    noise3= np.random.normal(0,error3,size = len(fs[3]))
    noise_ = np.array([noise0,noise1,noise2,noise3])
    
    if noise == True:
        return ts, fs+noise_, time_delays, noise_std,f,sample
    if noise == False:  
        return ts, fs, time_delays, noise_std,f,sample




# for i in range(4):
#     plt.plot(ts[i], fs[i]+noise[i], color='black', label = 'with noise')
#     plt.plot(ts[i], fs[i], color='orange', label='without noise')
#     plt.legend()
#     plt.xlabel('time[days]')
#     plt.ylabel('magnitude')
#     plt.show()

