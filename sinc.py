import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pickle

t_s = np.array(np.loadtxt('t_s.txt'))
with open('t_k_.pickle', 'rb') as g:
     t_k_ = pickle.load(g)
t_k = t_k_[0]
h = t_k_[1]
f_s = np.array(np.loadtxt('f.txt'))  # f(t_s)


#h = 3.0017678255745435
matrix = t_s[np.newaxis,:]-t_k[:,np.newaxis]
matrix_sinc = np.sinc(matrix/h)
transp = matrix_sinc.T
#print(transp)
#print(np.linalg.inv(transp))
f_k_T = np.linalg.lstsq(transp, f_s.T,rcond=-1)[0] # uses least squares
f_k = f_k_T.T # since its one dimensional, this should do nothing
#print(f_k,f_k_T)

# plt.plot(f_k@matrix_sinc,f_s,'o')
# plt.xlabel('f(t_k) x matrix_sinc')
# plt.ylabel('f(t_s)')
# plt.title('seing how the matrix solver performed')
# plt.plot(f_s,f_s,color='g',label='f(t_s) vs f(t_s)')
# plt.legend()
# plt.show()
cut = np.logical_and(f_k<np.max(f_s) , f_k>np.min(f_s))
plt.plot(t_s,f_s,'o')
plt.plot(t_k[cut],f_k[cut],'o')
plt.xlabel('time[days]')
plt.ylabel('magnitude')
plt.title('sinc interpolated values')
plt.show()
#print(f_k[f_k<2].shape,f_k.shape)
# change sampling, det(A.T @ A)-1
# gradient descent to minimize Y-AX
# take realistic time delays between 5 and 15 for ex
np.savetxt('matrix_sinc.txt',matrix_sinc)
np.savetxt('f_s.txt',f_s)
np.savetxt('f_k.txt',f_k)
print('f_k shape',f_s.shape)
print('f_k shape',f_k.shape)
