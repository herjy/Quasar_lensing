import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt('LightCurves_James/DRW015/LC_OBS_DRW015_NOI003_LSST-i.txt')


def get_epochs(S, sampling=0):

    n = S.size
    S_diff = np.abs(S - np.roll(S, -1, axis=1))
    plt.plot(S_diff.T);
    plt.show()
    if sampling == 0:
        sampling = np.median(S_diff, axis=1) * 3
    end_epoch = []
    for i in range(S_diff.shape[0]):
        print(i)
        end_epoch.append(np.where(S_diff[i, :] > sampling[i]))
    print(end_epoch)
    return end_epoch


class Quasar():
    def __init__(self, folder, filenames, bands):

        self.folder = folder
        self.filenames = filenames
        self.bands = bands
        self.data = None


    def get_data(self):
        data = []
        for band in self.bands:
            data.append([np.loadtxt(self.folder+'/'+self.filenames(band))])
        tab_data = np.array(data)
        sh = tab_data.shape
        data = np.reshape(tab_data, (sh[0],sh[2],sh[3]))
        return data

    def get_samplings(self):
        data = self.get_data()
        return data[:, :, 0]

    def get_LightCurves(self):
        data = self.get_data()
        return data[:,:,1]

    def get_errors(self):
        data = self.get_data()
        return data[:, :, 2]

    def plot_LightCurves(self, offset=0):
        S = self.get_samplings()
        L = self.get_LightCurves()
        E = self.get_errors()

        for i in range(len(self.bands)):
            plt.errorbar(S[i,:], L[i,:]+offset*i, yerr = E[i,:])
        plt.show()
        return None

    def get_epochs(s, sampling = 0):
        S
        n = S.size
        S_diff = np.abs(S-np.roll(S, -1, axis =1))
        plt.plot(S_diff.T); plt.show()
        if sampling == 0:
            sampling = np.median(S_diff, axis = 1)*3
        end_epoch = []
        for i in range(S_diff.shape[0]):

            end_epoch.append(np.where(S_diff[i,:]>sampling[i]))
        print(end_epoch)
        return end_epoch

    def full_sampling(self):
        S = self.get_samplings()
        epochs = self.get_epochs(self)

        for ep in epochs:
            if ep is not np.max(S):
                missing = np.linspace(S[ep], S[ep+1], S[ep+1]-S[ep]-1)
                S



def filename(i):
    return 'LC_OBS_DRW015_NOI003_LSST-'+i+'.txt'

Q = Quasar('LightCurves_James/DRW015', filename, ['u', 'g', 'r', 'i', 'z', 'y'])

Q.plot_LightCurves(offset = 3)
get_epochs(Q.get_samplings(), sampling = [1,1,1,1,1,1])
