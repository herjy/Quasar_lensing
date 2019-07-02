import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import *
from astroML.time_series import generate_damped_RW
import argparse
import random
import pickle

def index(dt_max,t_drive, num_curve=4):
    '''
    generates the time delays and outputs the indices of the individual time shifted samples within their sorted, concatenated array
    
    inputs:
    
    dt_max : the max of the generated time delay in days for ex: 15
    t_drive : the true time sample for ex: the array stored in t_drive_new.txt
    num_curve: the number of shifted curves we want to generate , default: 4
    
    output: time sample made of concatenated times of shifted curves , indices of each shifted curve within the sample array

    '''
    time_delay = np.random.rand(num_curve)*dt_max # generating num_curve time delays between 0 and dt_max
    
    # add t_drive to the concatenation to create the truth curve later on
    sample =np.concatenate((t_drive+time_delay[0],t_drive+time_delay[1],t_drive+time_delay[2],t_drive+time_delay[3]))#t_drive))
    

    n0 = len(t_drive+time_delay[0])
    n1 = len(t_drive+time_delay[1])
    n2 = len(t_drive+time_delay[2])
    n3 = len(t_drive+time_delay[3])
    ntruth = len(t_drive)

    index0 = []
    for i in range(n0):
        index0.append(np.where(np.argsort(sample) == i))
    index0 = np.array(index0).flatten()

    index1 = []
    for i in range(n0,n0+n1):
        index1.append(np.where(np.argsort(sample) == i))
    index1 = np.array(index1).flatten()

    index2 = []
    for i in range(n0+n1,n0+n1+n2):
        index2.append(np.where(np.argsort(sample) == i))
    index2 = np.array(index2).flatten()

    index3 = []
    for i in range(n0+n1+n2,n0+n1+n2+n3):
        index3.append(np.where(np.argsort(sample) == i))
    index3 = np.array(index3).flatten()


    index_truth = [] # only useful if t_drive is added to sample
    for i in range(n0+n1+n2+n3,n0+n1+n2+n3+ntruth):
        index_truth.append(np.where(np.argsort(sample) == i))
    index_truth = np.array(index_truth).flatten()
    
    return index0, index1, index2, index3, index_truth, sample,time_delay

# to check that index() is working, uncomment line below, replace 3 with 0,1,2,4
#np.sort(sample)[index3] == (t_drive+time_delay[3]) 


def simulation_(sample_):
    '''
    simulating data using any time sampling as the time axis

    input: 
    sample_: time sample made of concatenated arrays of the individual time sample of each shifted curve. default: sample obtained from index()
    
    output:
    magnitudes corresponding to the given time sample (--> full light curve.)

    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fname'         , '-fn  ', type=str  , default='LC_DRW.txt' ,help='Set the output filename (default: LC_DRW.txt)')
    parser.add_argument('--tau'           , '-tau ', type=float, default=300          ,help='Set the relaxation time (default: tau=300)')
    parser.add_argument('--strc_func_inf' , '-sf  ', type=float, default=0.1          ,help='Set the structure function at infinity (default: SFinf=0.1)')
    parser.add_argument('--xmean'         , '-xmu ', type=float, default=1.           ,help='Set the mean value of random walk (default: Xmean=1.)')
    parser.add_argument('--ran_seed'      , '-sn  ', type=str  , default='123'        ,help='Set the random seed (r: random, snnn: random seed)')
    parser.add_argument('--redshift_src'  , '-zs'  , type=float, default=0.5          ,help='Set the redshift of source (default: zs=0.5)')

    parser.add_argument('--target_dir'    , '-td'  ,             default='.',          help='Set the output directory')


    args = parser.parse_args()

    fn = args.fname

    xmean = args.xmean
    tau = args.tau
    SFinf = args.strc_func_inf
    zs = args.redshift_src
    sn = args.ran_seed

    stem_out = args.target_dir


    if (sn == 'r'):
        np.random.seed()
    else:
        sn = int(sn)
        np.random.seed(sn)
    sn = np.random.randn(1)

    # generating magnitudes

    return generate_damped_RW(np.sort(sample_), tau=tau, z=zs, SFinf=SFinf, xmean=xmean,random_state=5)

def plot(mag0,mag1,mag2,mag3, delay,t):
    '''
    plots the time shifter curves, plots alligned curves (shifted back)
    
    input:
    t : the original time sample of the true curve for ex. the array stored in time_drive.txt
    mag0,mag1,mag2,mag3 : magnitudes of the shifted curves, obtained from indexing the output of the simulation_ function
    delay : array containing the original time delays, obtained from the index_ function 
    
    output:
    plots of simulated data.
    '''
    
    #plt.plot(t_drive, f_truth, 'o', color='black', label='DRW- True')
    plt.plot(t, mag0+0.3, 'o', label='DRW1')  
    plt.plot(t, mag1+0.6, 'o',  label='DRW2')  
    plt.plot(t ,mag2+0.9, 'o', label='DRW3')  
    plt.plot(t, mag3+0.12, 'o', label='DRW4')  
    
    plt.title('time shifted curves')
    plt.xlabel('t (days)')
    plt.ylabel('Fraction Variation')
    plt.legend()
    plt.show()
    
    #plt.plot(t_drive, f_truth, color='black', label='DRW- True')
    plt.plot(t+delay[0], mag0, 'o', label='DRW1')  #+0.3
    plt.plot(t+delay[1], mag1, 'o',  label='DRW2')  #+0.6
    plt.plot(t+delay[2] ,mag2, 'o', label='DRW3')  #+0.9
    plt.plot(t+delay[3], mag3, 'o', label='DRW4')  #+0.12

    plt.title('shifting back and alligning all curves')
    plt.xlabel('t (days)')
    plt.ylabel('Fraction Variation')
    plt.legend()
    plt.show()
