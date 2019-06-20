import numpy as np
import matplotlib.pyplot as plt

#Loading the data
data = np.loadtxt('LC_DRW.txt')
t = data[:,0]
lc = np.copy(data[:,1])
t -= np.max(t)*2

#A draw of random time delays of the order of a couple days
dts = np.random.rand(4)*5
h = t[1]-t[0]
D = t[-1]-t[0]


print('minimum time interval: ',h)
print('duration in days:', D)
print('time delays:', dts)

#Limits of easonal gaps
#Set interupt to 0 to not have seasonal gaps
interupt = 1
interupt_lim = (np.array([200,325, 480, 605])/h).astype(int)


dtmin = dts.min()
dtmax = dts.max()
obs_lcs = []
#For each time delay
for dt in dts:
    #Time shift
    ct = t - dt
    #Tracks the time for which each light curve has a point
    min_lim = np.int((dt-dtmin)/h)
    if dt == dtmax:
        max_lim = None
    else:
        max_lim = np.int((dt-dtmax)/h)
    #Cuts the time shift to where each light curve has a point
    ct = ct[min_lim:max_lim]
    curve = lc[min_lim:max_lim]
    #Same for light curves. Also creat one light curve for each time delay
    obs_lcs.append(curve)



obs_lcs = np.array(obs_lcs)

#Times of observations designed to have one observation every 3 days on average
t_samp = (np.random.rand(np.int(D/3))*D+ct[0])

#Localisation of samples:
t_diff = np.abs(ct[:, np.newaxis]-t_samp[np.newaxis, :]).astype(int)
loc = np.argmin(t_diff, axis = 0)

#Makes the light curves a 4*N array
lcs_sampled = [obs_lcs[i][loc] for i in range(4)]


lcs_sampled = np.stack(lcs_sampled, axis = 0)

print(t_samp.shape, lcs_sampled.shape)

if interupt == 1:
    obs_lcs[:][interupt_lim[0]:interupt_lim[1]] = 0
    obs_lcs[:][interupt_lim[2]:interupt_lim[3]] = 0
#for i in range(4):
#    plt.plot(ct, obs_lcs[i]+np.random.rand(1)*obs_lcs[i].max(), label = 'Shifted observations')
plt.plot(t_samp, lcs_sampled[:].T+np.random.rand(4)*lcs_sampled[:].max()/2, 'o', label = 'Shifted observations')
plt.plot(t, data[:,1], 'r', label = 'Truth')
plt.show()

#Same plot as before but without magnitude shift
plt.plot(t_samp, lcs_sampled[:].T, 'o', label = 'Shifted observations')
plt.plot(t, data[:,1], 'r', label = 'Truth')
plt.show()