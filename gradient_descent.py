import numpy as np
import pickle 
import matplotlib.pyplot as plt

matrix_sinc = np.loadtxt('matrix_sinc.txt') # uploading the sinc matrix
f_s = np.loadtxt('f_s.txt') # f(t_s = current sampling)
f_k = np.loadtxt('f_k.txt') # f(t_k) as calculated by the least squares solution to use as an initial guess for gradient descent
t_s = np.array(np.loadtxt('t_s.txt')) # current sampling
with open('t_k_.pickle', 'rb') as g: # internet said to use this to open a tuple
     t_k_ = pickle.load(g)
t_k = t_k_[0]
#h = t_k_[1]

#first let's transpose:
A = matrix_sinc.T
Y = f_s.T

# now we want to solve Y = A.X using gradient descent

#let's calculate mu:
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
# linorm(A,100) found to be 0.20698332641517986 , higher iteration don't really change it, I copied it over to mu to avoid calculating it each time
mu = 0.20698*(1.5) 
# initialize X, I chose to use the sol of least squares as the initial guess (?)
X = f_k

while (np.linalg.norm(Y-A@X))**2 > 0.3: # calculated this norm^2 for the least squares sol it gave 0.5 so I chose 0.5>0.3 
    X_new = X + mu*((A.T)@(Y-A@X))
    X = X_new
#print(X == f_k)
# np.savetxt('f_k_gd',X)
#print(t_k)

cut = np.logical_and(X<np.max(f_s) , X>np.min(f_s)) # making a cut to avoid plotting outliers except that as of now all of them are outliers lol
plt.plot(t_s,f_s,'o',label = 'original')
plt.plot(t_k,X,'o',label='interpolated', color = 'orange') 
#plt.plot(t_k[cut],X[cut],'o',label='interpolated', color = 'orange')   
plt.xlabel('time[days]')
plt.ylabel('magnitude')
plt.title('sinc interpolated values')
plt.legend()
plt.show()
