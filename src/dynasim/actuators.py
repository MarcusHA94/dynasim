import numpy as np
import scipy
import scipy.signal
from math import pi



class excitation():

    def generate(self, time):
        return self._generate(time)
    
class impulse(excitation):
    '''
    Impulse excitation
    '''
    
    def __init__(self, imp_times = [0], width = 0.1, f0=0.1):
        self.imp_times = imp_times
        self.width = width
        self.f0 = f0
        
    def _generate(self, time, seed=43810):
        ns = time.shape[0]
        nt_width = int(self.width/(time[1]-time[0])) // 2
        f = np.zeros(ns)
        imp_locs = [np.argmin(np.abs(time - imp_time)) for imp_time in self.imp_times]
        for imp_loc in imp_locs:
            f[imp_loc - nt_width:imp_loc + nt_width] = self.f0
        return f
    
class sinusoid(excitation):
    '''
    Single sinusoidal signal with central frequency w, amplitude f0, and phase phi
    '''

    def __init__(self, w, f0=1.0, phi=0):
        self.w = w
        self.f0 = f0
        self.phi = phi

    def _generate(self, time, seed=43810):
        return self.f0 * np.sin(self.w*time + self.phi)
    
class white_gaussian(excitation):
    '''
    White Gaussian noise with variance f0, and mean
    '''

    def __init__(self, f0, mean=0.0):
        self.f0 = f0
        self.u = mean
    
    def _generate(self, time, seed=43810):
        ns = time.shape[0]
        np.random.seed(seed)
        return np.random.normal(self.u, self.f0*np.ones((ns)))
    
class banded_noise(excitation):

    def __init__(self, bandwidth, amplitude):
        self.bandwidth = bandwidth
        self.amplitude = amplitude

    def fftnoise(self, f):
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        return np.fft.ifft(f).real

    def band_limited_noise(self, min_freq, max_freq, samples=1024, samplerate=1):
        freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
        f[idx] = 1
        return self.fftnoise(f)

    def _generate(self, time, seed=43810):
        np.random.seed(seed)
        return self.amplitude * self.band_limited_noise(self.bandwidth[0], self.bandwidth[1], samples=len(time), samplerate=1/(time[1]-time[0]))
    
class sine_sweep(excitation):
    '''
    Sine sweep signal
    '''

    def __init__(self, w_l, w_u, F0=1.0, scale='linear'):
        self.w_l = w_l
        self.w_u = w_u
        self.F0 = F0
        self.scale = scale

    def _generate(self, time, seed=43810):
        f0 = self.w_l / (2*pi)
        f1 = self.w_u / (2*pi)
        F =  self.F0 * scipy.signal.chirp(time, f0, time[-1], f1, method=self.scale)
        return F
    
class rand_phase_ms(excitation):
    '''
    Random-phase multi-sine
    '''

    def __init__(self, freqs, Sx):
        self.freqs = freqs
        self.Sx = Sx

    def _generate(self, time, seed=43810):

        np.random.seed(seed)
        phases = np.random.randn(self.freqs.shape[0]) * pi/2
        F_mat = np.sin(time.reshape(-1,1) @ self.freqs.reshape(1,-1) + phases.T)

        return (F_mat @ self.Sx).reshape(-1)

class mdof_shaker():
    '''
    MDOF shaker class generates force signals at each DOF using excitation class
    '''

    def __init__(self, excitations=None, seed=43810):

        self.excitations = excitations
        self.dofs = len(excitations)
        self.seed = seed

    def generate(self, time):
        nt = time.shape[0]
        self.f = np.zeros((self.dofs,nt))
        for n, excite in enumerate(self.excitations):
            match excite:
                case excitation():
                    self.f[n, :] = self.excitations[n]._generate(time, self.seed+n)
                case np.ndarray():
                    self.f[n, :] = self.excitations[n]
                case None:
                    self.f[n, :] = np.zeros(nt)

        return self.f
    
class point_shakers():
    '''
    Point shakers class generates force signals at a specific locations using excitation class
    '''

    def __init__(self, excitations=None, xx=None, seed=43810):

        self.excitations = excitations
        "excitations is a list of dictionaries containing location and excitation"
        self.xx = xx
        self.seed = seed

    def generate(self, time):
        nt = time.shape[0]
        nx = self.xx.shape[0]
        self.f = np.zeros((nx, nt))
        for excite in self.excitations:
            self.loc_id = np.argmin(np.abs(self.xx-excite['loc']))
            match excite['excitation']:
                case excitation():
                    self.f[self.loc_id, :] = excite['excitation']._generate(time, self.seed)
                case np.ndarray():
                    self.f[self.loc_id, :] = self.excitation
        return self.f
        


