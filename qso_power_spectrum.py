import numpy as np
import matplotlib.pyplot as plt


numbers = []
for i in range(25):
    if i < 10:
        numbers.append('0'+str(i))
    else:
        numbers.append(str(i))

def file_name(n,m,b):
    return './LightCurves_James/DRW0'+n+'/LC_OBS_DRW0'+n+'_NOI0'+m+'_LSST-'+b+'.txt'

for num_folder in numbers:
#    for num_file in numbers:
    data = []
    for band in ['g','r','i','z','y']:
        data = np.loadtxt(file_name(num_folder, num_folder, band))
        time = data.T[0]
        mag = data.T[1]
        plt.psd(mag.astype(float), label = band)
        plt.legend()
    plt.show()




