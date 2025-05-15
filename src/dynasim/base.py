import numpy as np
from dynasim.simulators import *
from dynasim.actuators import mdof_shaker, point_shakers
import scipy.integrate as integrate
import warnings

class state_space_system:
    '''
    Base class for any state space represented system
    '''

    def __init__(self) -> int:

        return 0

    def gen_state_matrices(self) -> int:
        '''
        Generate state matrices A, H, and nonlinear matrix An based on system parameters
        '''

        self.A = np.concatenate((
            np.concatenate((np.zeros((self.dofs,self.dofs)), np.eye(self.dofs)), axis=1),
            np.concatenate((-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C), axis=1)
        ), axis=0)

        self.H = np.concatenate((np.zeros((self.dofs, self.dofs)), np.linalg.inv(self.M)), axis=0)

        match [self.Cn, self.Kn]:
            case [None,None]:
                self.An = None
            case [_,None]:
                self.An = np.concatenate((
                    np.zeros((self.dofs, 2*self.dofs)),
                    np.concatenate((np.zeros_like(self.Cn), -np.linalg.inv(self.M) @ self.Cn), axis=1)
                ), axis=0)
            case [None, _]:
                self.An = np.concatenate((
                    np.zeros((self.dofs, 2*self.dofs)),
                    np.concatenate((-np.linalg.inv(self.M) @ self.Kn, np.zeros_like(self.Kn)), axis=1)
                ), axis=0)
            case [_,_]:
                self.An = np.concatenate((
                    np.zeros((self.dofs, 2*self.dofs)),
                    np.concatenate((-np.linalg.inv(self.M) @ self.Kn, -np.linalg.inv(self.M) @ self.Cn), axis=1)
                ), axis=0)
        
        return 0
    
    def gen_obs_matrices(self) -> int:
        '''
        Generate observation matrices B and D based on system parameters
        '''
        
        self.B = np.concatenate((-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C), axis=1)
        self.D = np.linalg.inv(self.M)

        match [self.Cn, self.Kn]:
            case [None,None]:
                self.Bn = None
            case [_, None]:
                self.Bn = np.concatenate((np.zeros_like(self.Cn), -np.linalg.inv(self.M) @ self.Cn), axis=1)
            case [None, _]:
                self.Bn = np.concatenate((-np.linalg.inv(self.M) @ self.Kn, np.zeros_like(self.Kn)), axis=1)
            case [_,_]:
                self.Bn = np.concatenate((-np.linalg.inv(self.M) @ self.Kn, -np.linalg.inv(self.M) @ self.Cn), axis=1)

        return 0
    
class mdof_system(state_space_system):
    '''
    Base class for generic mdof system
    '''

    def __init__(self, M=None, C=None, K=None, Cn=None, Kn=None, nonlinearity=None):

        self.M = M
        self.C = C
        self.K = K
        self.Cn = Cn
        self.Kn = Kn
        self.dofs = M.shape[0]
        self.nonlinearity = nonlinearity

        self.gen_state_matrices()
        self.gen_obs_matrices()
    
    def nonlin_transform(self, z):

        if self.nonlinearity is not None:

            x_ = z[:self.dofs] - np.concatenate((np.zeros_like(z[:1]), z[:self.dofs-1]))
            x_dot = z[self.dofs:] - np.concatenate((np.zeros_like(z[:1]), z[self.dofs:-1]))

            return np.concatenate((
                self.nonlinearity.gk_func(x_, x_dot),
                self.nonlinearity.gc_func(x_, x_dot)
            ))
        
        else:
            return np.zeros_like(z)

    def simulate(self, tt, z0=None, simulator=None):
        '''
        Simulate the system for a given time using the specified simulator.

        Args:
            tt: The vector of time samples
            z0: The initial state of the system. Defaults to None.
            simulator: The simulator to use for the simulation. Defaults to scipy.solve_ivp.

        Returns:
            A dictionary containing the system's response displacement and velocity over time.
        '''

        # try:
        # instantiate time
        self.t = tt

        if hasattr(self, 'excitations'):
            # create shaker object
            self.shaker = mdof_shaker(self.excitations)
            # generate forcing series
            self.f = self.shaker.generate(tt)
        else:
            self.f = None

        # initiate simulator
        if simulator is None:
            # self.simulator = scipy_ivp(self)
            self.simulator = rk4(self)
        else:
            self.simulator = simulator(self)

        # initial conditions
        if z0 is None:
            # warnings.warn('No initial conditions provided, proceeding with zero initial state', UserWarning)
            z0 = np.zeros((2*self.dofs))
            if all([e is None for e in self.excitations]):
                warnings.warn('Zero initial condition and zero excitations, what do you want??', UserWarning)
        
        # simulate
        simulated_state_data = self.simulator.sim(tt, z0)
        
        # generate acceleration observation data
        z = simulated_state_data['z']
        if not self.simulator.__class__.__name__ == 'corotational_rk4':
            if self.Bn is None:
                acceleration = self.B @ z + self.D @ self.f
            else:
                acceleration = self.B @ z + self.Bn @ self.nonlin_transform(z) + self.D @ self.f
                
            simulated_state_data.update({'acc': acceleration})
        
        # Final debug print to check the updated dictionary
        # print("Updated simulated_state_data:", simulated_state_data)
            
        # except Exception as e:
        #     print("An error occurred during simulation:", e)
        #     simulated_state_data = None
            
        return simulated_state_data

class cont_ss_system(state_space_system):
    '''
    Base class for continuous system represented in the state space
    '''

    def __init__(self, M=None, C=None, K=None, modes=1):

        self.M = M
        self.C = C
        self.K = K
        self.Cn = None  # no nonlinearities for now
        self.Kn = None  # no nonlinearities for now
        self.dofs = modes

        self.gen_state_matrices
    
    def simulate(self, tt, z0=None, simulator=None):
        '''
        Simulate the system for a given time using the specified simulator.

        Args:
            tt: The vector of time samples
            z0: The initial state of the system. Defaults to None.
            simulator: The simulator to use for the simulation. Defaults to scipy.solve_ivp.

        Returns:
            A dictionary containing the system's response displacement and velocity over time.
        '''

        # instantiate time
        self.t = tt

        if hasattr(self, 'excitations'):
            if self.excitations is not None:
                # create shaker object
                self.shaker = point_shakers(self.excitations, self.xx)
                # generate forcing series
                # self.f = self.shaker.generate(tt)
                ff = self.shaker.generate(tt)
                pp = np.zeros((self.n_modes, tt.shape[0]))
                for n in range(self.n_modes):
                    pp[n,:] = integrate.simpson(self.phi_n[:,n].reshape(-1,1) * ff, self.xx, axis=0).reshape(-1)
                self.f = pp
            else:
                self.f = None
        else:
            self.f = None

        # initiate simulator
        if simulator is None:
            # self.simulator = scipy_ivp(self)
            self.simulator = rk4(self)
        else:
            self.simulator = simulator(self)

        # initial conditions
        if z0 is None:
            # warnings.warn('No initial conditions provided, proceeding with zero initial state', UserWarning)
            tau0 = np.zeros((2*self.dofs))
            if all([e is None for e in self.excitations]):
                warnings.warn('Zero initial condition and zero excitations, what do you want??', UserWarning)
        else:
            tau0 = z0
        
        # simulate
        qq = self.simulator.sim(tt, tau0)
        return qq





        