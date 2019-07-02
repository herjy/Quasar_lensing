import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def T_K(step,sample):
    '''
    input:
    step : the time step for the desired sampling
    output:
    returns a tuple: 0th index is the time sampling and 1st index is the time step (approximately the same as the argument 'step')
    '''
    
    min_ = np.ceil(np.min(sample))
    max_ = np.floor(np.max(sample))
    
    return np.linspace(min_,max_,np.int(np.abs((min_-max_)/step)),dtype=int,retstep=True)

def least_squares(t_s,T_K,f_s):
    ''' 
    output:
    the sinc matrix made of sinc(tk-ts /h)
    plot of interpolated values using sinc and least squares 
    '''
    t_k = T_K[0]
    h = T_K[1]
    matrix = t_s[np.newaxis,:]-t_k[:,np.newaxis]
    matrix_sinc = np.sinc(matrix/h)
    transp = matrix_sinc.T
    f_k_T = np.linalg.lstsq(transp, f_s.T,rcond=-1)[0] # uses least squares
    f_k = f_k_T.T # since its one dimensional, this should do nothing
    
    #PLOT:
    
    cut = np.logical_and(f_k<np.max(f_s) , f_k>np.min(f_s))
    plt.plot(t_s,f_s,'o')
    plt.plot(t_k[cut],f_k[cut],'o')
    plt.xlabel('time[days]')
    plt.ylabel('magnitude')
    plt.title('sinc interpolated values')
    plt.show()
        


def gradient_descent(t_s,T_K,f_s):
    t_k = T_K[0]
    h = T_K[1]
    matrix = t_s[np.newaxis,:]-t_k[:,np.newaxis]
    matrix_sinc = np.sinc(matrix / h)
    A = matrix_sinc.T
    Y = f_s.T
    
    def linorm(S, nit):
        n1, n2 = np.shape(S)
        x0 = np.random.rand(1, n1)
        x0 = x0 / np.sqrt(np.sum(x0 ** 2))    
        for i in range(nit):
            x = np.dot(x0, S)
            xn = np.sqrt(np.sum(x ** 2))
            xp = x / xn
            y = np.dot(xp, S.T)
            yn = np.sqrt(np.sum(y ** 2))       
            if yn < np.dot(y, x0.T):
                break
            x0 = y / yn
        return 1./xn
    
    mu = linorm(A, 20)/100.
    
    X = np.zeros(len(t_k))
    
    count = 0
    R = [np.sum(Y**2)]
    epsilon = 0.3
    
    while (R[-1] > epsilon) and (count < 10000): # calculated this norm^2 for the least squares sol it gave 0.5 so I chose 0.5>0.3
        X_new = X + mu*((A.T)@(Y-A@X))
        X = X_new.copy()
        R.append(np.sum((Y-A@X)**2.))
        count+=1
        
    # convergence plot 
    
    plt.title('Convergence')
    plt.plot(np.array(R[1:]))
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.show()
    
    # interpolated values plot
    cut = np.logical_and(X < np.max(f_s), X > np.min(f_s))
    plt.plot(t_s,f_s,'o',label = 'original')
    plt.plot(t_k[cut],X[cut],'o',label='interpolated', color = 'orange')
    plt.xlabel('time[days]')
    plt.ylabel('magnitude')
    plt.title('sinc interpolated values')
    plt.legend()
    plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    