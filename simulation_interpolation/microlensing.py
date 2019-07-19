import numpy as np



def make_microlensing(t= np.loadtxt('t_drive_new.txt'), u0=1.3, te=500, d=0):
    '''
    input:
    t = time axis array , default: t_drive_new.txt
    u0 =  closest projected approach : controls the height of the peak, the large u0 the
    te = time it takes the source to travel one Einstein radius : controls the width of the peak
    d: to control the time from the middle of t to the time at the peak: d=0 the peak is at the center of t
    output:
    magnitudes of microlensing effect at the given times t 
    amplitude of the graph 
    '''
    t0 = (np.max(t)+np.min(t))/2 + d

    def u(t):
        return np.sqrt(u0**2 +  ((t-t0)/te)**2 )   
    def A(u):
        return (u**2+2)/(u*np.sqrt(u**2 + 4))
    
    mag = A(u(t))-1

    return t,mag ,np.min(mag)

# to plot 
# mic = make_microlensing()
# plt.plot(mic[0],mic[1],'o')
# plt.show()
