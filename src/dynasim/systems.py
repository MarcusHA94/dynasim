import numpy as np
from math import pi
from dynasim.base import mdof_system, cont_ss_system
import scipy.integrate as integrate

class cont_beam(cont_ss_system):

    def __init__(self, def_type, **kwargs):
        super().__init__()

        self.mat_def_type = def_type
        self.L = kwargs["l"]
        self.nonlin_transform = lambda z : np.zeros_like(z)

        match def_type:
            case "full_vars":
                self.E = kwargs["E"]
                self.rho = kwargs["rho"]
                self.I = kwargs["I"]
                if isinstance(kwargs["area"], list) or isinstance(kwargs["area"], tuple):
                    self.b = kwargs["area"][0]
                    self.h = kwargs["area"][1]
                    self. A = np.product(kwargs["area"])
                    if np.abs(self.I - (1/12)*self.b*self.h**3)/self.I > 0.01:
                        raise ValueError("Moment of inertia does not match values of b and h...")
                else:
                    self.A = kwargs["area"]
                self.c = kwargs["c"]
                self.pA = self.rho * self.A
            case "cmb_vars":
                self.EI = kwargs["EI"]
                self.pA = kwargs["pA"]
                self.c = kwargs["c"]
            case "uni_vars":
                self.mu = kwargs["mu"]
                self.c = kwargs["c"]
        
    def gen_modes(self, bc_type, n_modes, nx):

        self.bc_type = bc_type
        self.nx = nx
        x = np.linspace(0, self.L, nx)
        self.xx = x
        self.n_modes = n_modes
        self.dofs = n_modes
        nn = np.arange(1, n_modes+1, 1)
        match self.mat_def_type:
            case "full_vars":
                wn_mult = (self.E * self.I / (self.rho * self.A * self.L**4))**(0.5)
            case "cmb_vars":
                wn_mult = (self.EI / (self.pA * self.L**4))**(0.5)
            case "uni_vars":
                wn_mult = (self.mu / (self.L**4))**(0.5)
                self.pA = 1.0

        match bc_type:
            case "ss-ss":
                Cn = np.sqrt((2/(self.pA*self.L)))
                self.bc_type_long = "simply supported - simply supported"
                beta_l = nn*pi
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = np.zeros((self.nx, n_modes))
                self.phi_dx2_n = np.zeros((self.nx, n_modes))
                self.phi_dx4_n = np.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    beta_n = beta_l[n]/self.L
                    self.phi_n[:,n] = Cn * np.sin(beta_n*x)
                    self.phi_dx2_n[:,n] = -Cn * (beta_n**2)*np.sin(beta_n*x)
                    self.phi_dx4_n[:,n] = Cn * (beta_n**4)*np.sin(beta_n*x)
            case "fx-fx":
                self.bc_type_long = "fixed - fixed"
                beta_l = (2*nn + 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = np.zeros((self.nx, n_modes))
                self.phi_dx2_n = np.zeros((self.nx, n_modes))
                self.phi_dx4_n = np.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) - np.cosh(beta_l[n]))/(np.sin(beta_l[n]) - np.sinh(beta_l[n])) * \
                                    (np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    self.phi_dx2_n[:,n] =  beta_n[n]**2 * (-np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**2 * (-np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    self.phi_dx4_n[:,n] = beta_n[n]**4 * (np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**4 * (np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
            case "fr-fr":
                self.bc_type_long = "free - free"
                beta_l = (2*nn + 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = np.zeros((self.nx, n_modes))
                self.phi_dx2_n = np.zeros((self.nx, n_modes))
                self.phi_dx4_n = np.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (np.cos(beta_n[n]*x) + np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) - np.cosh(beta_l[n]))/(np.sin(beta_l[n]) - np.sinh(beta_l[n])) * \
                                    (np.sin(beta_n[n]*x) + np.sinh(beta_n[n]*x))
                    self.phi_dx2_n[:,n] =  beta_n[n]**2 * (-np.cos(beta_n[n]*x) + np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**2 * (-np.sin(beta_n[n]*x) + np.sinh(beta_n[n]*x))
                    self.phi_dx4_n[:,n] = beta_n[n]**4 * (np.cos(beta_n[n]*x) + np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**4 * (np.sin(beta_n[n]*x) + np.sinh(beta_n[n]*x))
            case "fx-ss":
                self.bc_type_long = "fixed - simply supported"
                beta_l = (4*nn + 1) * pi / 4
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = np.zeros((self.nx, n_modes))
                self.phi_dx2_n = np.zeros((self.nx, n_modes))
                self.phi_dx4_n = np.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) - np.cosh(beta_l[n]))/(np.sin(beta_l[n]) - np.sinh(beta_l[n])) * \
                                    (np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    self.phi_dx2_n[:,n] =  beta_n[n]**2 * (-np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**2 * (-np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    self.phi_dx4_n[:,n] = beta_n[n]**4 * (np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**4 * (np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
            case "fx-fr":
                self.bc_type_long = "fixed - free"
                beta_l = (2*nn - 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = np.zeros((self.nx, n_modes))
                self.phi_dx2_n = np.zeros((self.nx, n_modes))
                self.phi_dx4_n = np.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    (np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    self.phi_dx2_n[:,n] =  beta_n[n]**2 * (-np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**2 * (-np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    self.phi_dx4_n[:,n] = beta_n[n]**4 * (np.cos(beta_n[n]*x) - np.cosh(beta_n[n]*x)) - \
                                    (np.cos(beta_l[n]) + np.cosh(beta_l[n]))/(np.sin(beta_l[n]) + np.sinh(beta_l[n])) * \
                                    beta_n[n]**4 * (np.sin(beta_n[n]*x) - np.sinh(beta_n[n]*x))
                    
        M = np.zeros((self.n_modes,self.n_modes))
        K = np.zeros((self.n_modes,self.n_modes))
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                m_integrand = self.phi_n[:,i].reshape(-1,1) * self.phi_n[:,j].reshape(-1,1)
                M[i,j] = integrate.simpson(m_integrand.reshape(-1),self.xx)
                k_integrand = self.phi_dx2_n[:,i].reshape(-1,1) * self.phi_dx2_n[:,j].reshape(-1,1)
                K[i,j] = integrate.simpson(k_integrand.reshape(-1),self.xx)
        self.M = self.pA * M
        self.C = self.pA * self.c * M
        self.K = self.EI * K

        self.gen_state_matrices()

        return self.xx, self.phi_n


class mdof_symmetric(mdof_system):
    '''
    Generic MDOF symmetric system
    '''

    def __init__(self, m_, c_, k_, dofs=None, nonlinearity=None):

        if type(m_) is np.ndarray:
            dofs = m_.shape[0]
        elif dofs is not None:
            m_ = m_ * np.ones((dofs))
            c_ = c_ * np.ones((dofs))
            k_ = k_ * np.ones((dofs))
        else:
            raise Exception('Under defined system, please provide either parameter vectors or number of degrees of freedom')
        
        self.m_ = m_
        self.c_ = c_
        self.k_ = k_

        M = np.diag(m_) 
        C = np.diag(c_[:-1]+c_[1:]) + np.diag(-c_[1:],k=1) + np.diag(-c_[1:],k=-1)
        K = np.diag(k_[:-1]+k_[1:]) + np.diag(-k_[1:],k=1) + np.diag(-k_[1:],k=-1)

        if nonlinearity is not None:
            self.nonlin_transform = lambda z : nonlinearity.z_func(
                np.concatenate((z[:dofs],2*z[dofs-1])) - np.concatenate((np.zeros_like(z[:1]),z[:dofs])),
                np.concatenate((z[dofs:],2*z[-1])) - np.concatenate((np.zeros_like(z[:1]),z[dofs:]))
                )
            Cn = nonlinearity.Cn
            Kn = nonlinearity.Kn
            super().__init__(M, C, K, Cn, Kn)
        else:
            super().__init__(M, C, K)


class mdof_cantilever(mdof_system):
    '''
    Generic MDOF "cantilever" system
    '''

    def __init__(self, m_, c_, k_, dofs=None, nonlinearity=None):

        if type(m_) is np.ndarray:
            dofs = m_.shape[0]
        elif dofs is not None:
            m_ = m_ * np.ones((dofs))
            c_ = c_ * np.ones((dofs))
            k_ = k_ * np.ones((dofs))
        else:
            raise Exception('Under defined system, please provide either parameter vectors or number of degrees of freedom')
        
        self.m_ = m_
        self.c_ = c_
        self.k_ = k_
        self.nonlinearity = nonlinearity

        M = np.diag(m_)
        C = np.diag(np.concatenate((c_[:-1]+c_[1:],np.array([c_[-1]])),axis=0)) + np.diag(-c_[1:],k=1) + np.diag(-c_[1:],k=-1)
        K = np.diag(np.concatenate((k_[:-1]+k_[1:],np.array([k_[-1]])),axis=0)) + np.diag(-k_[1:],k=1) + np.diag(-k_[1:],k=-1)

        if nonlinearity is not None:
            Cn = nonlinearity.Cn
            Kn = nonlinearity.Kn
            super().__init__(M, C, K, Cn, Kn)
        else:
            super().__init__(M, C, K)
    
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
        

class bid_mdof_walled(mdof_system):
    '''
    Generic bidirection MDOF system with walls
    '''
    
    def __init__(self, mm, cc_h, cc_v, kk_h, kk_v, shape=None):
        
        if type(mm) is np.ndarray:
            dofs = 2 * mm.size
        elif dofs is not None:
            dofs = 2 * shape[0] * shape[1]
            mm = mm * np.ones(shape)
            cc_h = cc_h * np.ones(shape)
            cc_v = cc_v * np.ones(shape)
            kk_h = kk_h * np.ones(shape)
            kk_v = kk_v * np.ones(shape)
        else:
            raise Exception('Under defined system, please provide either parameter matrices or number of degrees of freedom and shape')
        
        self.mm_ = mm
        self.cc_h = cc_h
        self.cc_v = cc_v
        self.kk_h = kk_h
        self.kk_v = kk_v
        
        M = np.diag(mm.reshape(-1).repeat(2))
        C = self.construct_modal_matrix(cc_h, cc_v)
        K = self.construct_modal_matrix(kk_h, kk_v)
        
        self.nonlinearity = None
        
        super().__init__(M, C, K)
    
    
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
    
    def stack_connection_x(self, ck_dir):
        
        m, n = ck_dir.shape
        CK_dir = np.zeros((m * n, m * n))
        for i in range(m):
            ck_i = ck_dir[i, :]
            CK_i = np.diag(np.concatenate((ck_i[:-1]+ck_i[1:],np.array([ck_i[-1]])),axis=0)) + np.diag(-ck_i[1:],k=1) + np.diag(-ck_i[1:],k=-1)
            CK_dir[i*n:(i+1)*n, i*n:(i+1)*n] = CK_i
        return CK_dir
    
    def stack_connection_y(self, ck_dir):
        
        m, n = ck_dir.shape
        CK_dir = np.zeros((m * n, m * n))
        
        off_diag_vec = -np.concatenate([ck_dir[i, :] for i in range(1,m)])
        CK_dir = CK_dir + np.diag(off_diag_vec, k=n)
        CK_dir = CK_dir + np.diag(off_diag_vec, k=-n)
        
        diag_vec = np.concatenate([ck_dir[i, :] + ck_dir[i+1, :] for i in range(m-1)] + [ck_dir[-1, :]])
        
        CK_dir = CK_dir + np.diag(diag_vec)
        
        return CK_dir
        
    def construct_modal_matrix(self, ck_h, ck_v):
        '''
        Construct the modal matrix for the system
        '''
        # Get dimensions of ck_h and ck_v (assuming both are m x n and the same size)
        m, n = ck_h.shape
        
        # Initialize a (2 * m) x (2 * n) zero matrix for CK
        CK = np.zeros((2 * m * n, 2 * m * n))

        def index(i, j, direction):
            # Helper function to calculate the position in the stiffness matrix
            # direction = 0 for x, 1 for y
            return 2 * (i * n + j) + direction
        
        CK_x = self.stack_connection_x(ck_h)
        CK_y = self.stack_connection_y(ck_v)
        
        CK[::2, ::2] = CK_x
        CK[1::2, 1::2] = CK_y

        return CK
        