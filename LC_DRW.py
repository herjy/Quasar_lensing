#! /usr/bin/env python

import numpy as np
from math import *

from astropy.cosmology import LambdaCDM
from astropy.constants import *
from astropy import units

from scipy import signal

from astroML.time_series import generate_damped_RW

import pyfits 
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import time
import argparse

# e.g.
# LC_DRW.py -tau 90 -sf 0.2 -xmu 1. -sn r

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--time_initial'  , '-ti  ', type=float, default=-365.        ,help='Set the initial time (default: t=-365 [day])')
parser.add_argument('--time_final'    , '-tf  ', type=float, default=365.         ,help='Set the final time (default: t=365 [day])')
parser.add_argument('--delta_time'    , '-dt  ', type=float, default=0.1          ,help='Set the time interval (default: dt=0.1 [day])')
parser.add_argument('--fname'         , '-fn  ', type=str  , default='LC_DRW.txt' ,help='Set the output filename (default: LC_DRW.txt)')

parser.add_argument('--tau'           , '-tau ', type=float, default=300          ,help='Set the relaxation time (default: tau=300)')
parser.add_argument('--strc_func_inf' , '-sf  ', type=float, default=0.1          ,help='Set the structure function at infinity (default: SFinf=0.1)')
parser.add_argument('--xmean'         , '-xmu ', type=float, default=1.           ,help='Set the mean value of random walk (default: Xmean=1.)')
parser.add_argument('--ran_seed'      , '-sn  ', type=str  , default='123'        ,help='Set the random seed (r: random, snnn: random seed)')
parser.add_argument('--redshift_src'  , '-zs'  , type=float, default=0.5          ,help='Set the redshift of source (default: zs=0.5)')

parser.add_argument('--target_dir'    , '-td'  ,             default='.',          help='Set the output directory')


args = parser.parse_args()

ti = args.time_initial
tf = args.time_final
dt = args.delta_time
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

###########################################


########################
#      light curve     #
########################

for i in range(1000):
    sn = np.random.randn(1)

    print('-------------------------------')
    print('       light curve (DRW)       ')
    print('-------------------------------')
    print('Xmean =',xmean)
    print('tau   =',tau,'day')
    print('zs    =',zs)
    print('SFinf =','{0:.3f}'.format(SFinf))
    print('sn    =',sn)

    #t_drive = np.arange(ti, tf, dt)
    t_drive = np.arange(ti, tf+dt, dt)
    f_drive = generate_damped_RW(t_drive, tau=tau, z=zs, SFinf=SFinf, xmean=xmean)

    f_drive = abs(f_drive)
    mean = np.mean(f_drive)
    std  = np.std( f_drive)

    print('Light Curve:','from',ti,'to',tf,'days')
    print('mean =','{0:.3f}'.format(mean),'std =','{0:.3f}'.format(std))

    fn = stem_out+'/'+fn
    #np.savetxt(fn,zip(t_drive, f_drive),fmt='%f %f')

    plt.plot(t_drive, f_drive,  color='black', label='DRW')

    plt.xlabel('t (days)')
    plt.ylabel('Fraction Variation')
    plt.legend(loc=3)
    plt.show()
