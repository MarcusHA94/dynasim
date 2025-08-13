import numpy as np
from math import pi
from dynasim.base import mdof_system, cont_ss_system
import scipy.integrate as integrate
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import Delaunay

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
            super().__init__(M, C, K, Cn, Kn, nonlinearity)
        else:
            super().__init__(M, C, K)
    
    def nonlin_transform(self, z):

        if self.nonlinearity is not None:

            x_ = z[:self.dofs] - np.concatenate((np.zeros_like(z[:1]), z[:self.dofs]))
            x_dot = z[self.dofs:] - np.concatenate((np.zeros_like(z[:1]), z[self.dofs:]))

            return np.concatenate((
                self.nonlinearity.gk_func(x_, x_dot),
                self.nonlinearity.gc_func(x_, x_dot)
            ))
        
        else:
            return np.zeros_like(z)
        

class grid_uncoupled(mdof_system):
    '''
    Generic bidirection MDOF system with walls and uncoupled x and y motion (i.e. linearised by assuming small angles)
    '''
    
    def __init__(self, mm, cc_h, cc_v, kk_h, kk_v, shape=None, nonlinearity=None):
        
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
        K = self.construct_modal_matrix(kk_h, kk_v)
        C = self.construct_modal_matrix(cc_h, cc_v)
        
        self.shape = shape
        
        self.nonlinearity = nonlinearity
        
        if nonlinearity is not None:
            Cn = nonlinearity.Cn
            Kn = nonlinearity.Kn
            super().__init__(M, C, K, Cn, Kn)
        else:
            super().__init__(M, C, K)
    
    def nonlin_transform(self, z):

        if self.nonlinearity is not None:
            
            m, n = self.shape
            if len (z.shape) == 1:
                z = z.reshape(-1, 1)
            nt = z.shape[1]
            
            x_ = np.zeros_like(z[:self.dofs, :])
            x_dot = np.zeros_like(z[self.dofs:, :])
            
            for t in range(nt):
            
                displacement = z[:self.dofs, t].reshape(m, n, 2)
                result = []

                for i in range(m):
                    for j in range(n):
                        # x-displacement
                        if j == 0:  # First column
                            result.append(displacement[i, j, 0])
                        else:  # Difference with the previous x
                            result.append(displacement[i, j, 0] - displacement[i, j - 1, 0])

                        # y-displacement
                        if i == 0:  # First row
                            result.append(displacement[i, j, 1])
                        else:  # Difference with the previous y
                            result.append(displacement[i, j, 1] - displacement[i - 1, j, 1])

                x_[:, t] = np.array(result)
            
            return np.concatenate((
                self.nonlinearity.gk_func(x_, x_dot),
                self.nonlinearity.gc_func(x_, x_dot)
            ))
        
        else:
            return np.zeros_like(z)
        
    def construct_modal_matrix(self, ck_h, ck_v):
        '''
        Construct the modal matrix for the system
        '''
        # Get dimensions of ck_h and ck_v (assuming both are m x n and the same size)
        m, n = ck_h.shape
        
        num_dof = 2 * m * n
        
        # Initialize a (2 * m) x (2 * n) zero matrix for CK
        CK = np.zeros((num_dof, num_dof))
        
        def index(i, j, direction):
            return 2 * (i * n + j) + direction # 0 for horizontal, 1 for vertical
        
        # populate the stiffness matrix
        for i in range(m):
            for j in range(n):
                x_idx = index(i, j, 0)  # index for x^{i,j}
                y_idx = index(i, j, 1)  # index for y^{i,j}
                    
                # contribution to left neighbour x^{i,j-1}
                if j > 0:
                    left_idx = index(i, j-1, 0)
                    CK[left_idx, x_idx] -= ck_h[i, j]
                    
                # contribution to right neighbour x^{i,j+1}
                if j < n-1:
                    right_idx = index(i, j+1, 0)
                    CK[right_idx, x_idx] -= ck_h[i, j]
                    
                # contribution to top neighbour y^{i-1,j}
                if i > 0:
                    top_idx = index(i-1, j, 1)
                    CK[top_idx, y_idx] -= ck_v[i, j]
                    
                # contribution to bottom neighbour y^{i+1,j}
                if i < m-1:
                    bottom_idx = index(i+1, j, 1)
                    CK[bottom_idx, y_idx] -= ck_v[i, j]
                    
                # contribution to self
                CK[x_idx, x_idx] += ck_h[i, j]
                CK[y_idx, y_idx] += ck_v[i, j]
                if j < n-1:
                    CK[x_idx, x_idx] += ck_h[i, j+1]
                if i < m-1:
                    CK[y_idx, y_idx] += ck_v[i+1, j]
        
        # turn lower triangle into zeros
        CK = np.triu(CK)
        
        # replace lower triangle with upper triangle
        CK = CK + CK.T - np.diag(np.diag(CK))
                

        return CK
    
class grid_corotational(mdof_system):
    """
    Same input signature as *grid_uncoupled* but bars are assembled
    with a co-rotational (large-angle) stiffness & damping every step.
    """
    
    EPS_len = 1e-9  # minimum bar length used in any 1/L
    S_MAX = 1e12  # optional force clipping

    # ---------- constructor ----------------------------------------------
    def __init__(self, mm, cc_h, cc_v, kk_h, kk_v,
                 shape=None, nonlinearity=None, sparse=True):

        # --------- interpret inputs exactly like grid_uncoupled -----------
        if isinstance(mm, np.ndarray):
            self.mm_   = mm
            self.cc_h  = cc_h
            self.cc_v  = cc_v
            self.kk_h  = kk_h
            self.kk_v  = kk_v
            self.kn_h  = nonlinearity.kn_h if nonlinearity is not None else np.zeros_like(kk_h)
            self.kn_v  = nonlinearity.kn_v if nonlinearity is not None else np.zeros_like(kk_v)
            m, n       = mm.shape
        elif shape is not None:      # scalar → fill arrays
            m, n       = shape
            self.mm_   = mm   * np.ones(shape)
            self.cc_h  = cc_h * np.ones(shape)
            self.cc_v  = cc_v * np.ones(shape)
            self.kk_h  = kk_h * np.ones(shape)
            self.kk_v  = kk_v * np.ones(shape)
            self.kn_h  = nonlinearity.kn_h if nonlinearity is not None else np.zeros(shape)
            self.kn_v  = nonlinearity.kn_v if nonlinearity is not None else np.zeros(shape)
        else:
            raise ValueError("Either provide full arrays or the 'shape'.")
        
        self.nonlinearity = nonlinearity

        self.shape  = (m, n)
        self.sparse = sparse

        # ---------- constant mass matrix (2 N × 2 N) ----------------------
        M = np.diag(self.mm_.reshape(-1).repeat(2))

        # ---------- build node coordinates --------------------------------
        # (unit spacing; origin top-left, like the uncoupled code)
        xv, yv      = np.meshgrid(np.arange(n), np.arange(m))   # shape (m,n)
        self.node_coords = np.column_stack((xv.ravel(), yv.ravel()))
        self.N      = self.node_coords.shape[0]                 # = m*n

        # ---------- build bar list & per-bar constants --------------------
        self.bars, self.k_e, self.c_e, self.kn_e = self._make_bar_arrays()
        
        # --------- generate L0_e (rest length) from bar coordinates ---------
        self.L0_e = np.hypot(*(self.node_coords[self.bars[:,1]] -
                       self.node_coords[self.bars[:,0]]).T)

        # ---------- mdof_system needs *some* K,C for bookkeeping ----------
        # pass zeros; they are never used in the co-rotational simulator
        Z = np.zeros((2*self.N, 2*self.N))
        super().__init__(M, Z, Z, nonlinearity=nonlinearity)
        
        # ---- local (2-node, 4-dof) axial bar matrices ----
        self._k_axial = np.array([[ 1, 0, -1, 0],
                            [ 0, 0,  0, 0],
                            [-1, 0,  1, 0],
                            [ 0, 0,  0, 0]], dtype=float)

        self._c_axial = self._k_axial.copy()           # exactly the same pattern
        
        self.k_local_scaled = (self._k_axial[None,:,:] * self.k_e[:,None,None])
        self.c_local_scaled = (self._c_axial[None,:,:] * self.c_e[:,None,None])

    # ---------------------------------------------------------------------
    def _make_bar_arrays(self):
        """Generate connectivity arrays from kk_h/kk_v & cc_h/cc_v."""
        bars = []
        k_e  = []
        c_e  = []
        kn_e = []
        m, n = self.shape

        # helper to convert (i,j) → global node index
        node = lambda i, j: i * n + j

        # -------- horizontal bars (between (i,j-1) ↔ (i,j)) --------------
        for i in range(m):
            for j in range(1, n):                        # skip j=0 wall entry
                p, q = node(i, j-1), node(i, j)
                bars.append((p, q))
                k_e.append(self.kk_h[i, j])
                c_e.append(self.cc_h[i, j])
                kn_e.append(self.kn_h[i, j])

        # -------- vertical bars (between (i-1,j) ↔ (i,j)) ----------------
        for i in range(1, m):                            # skip top wall row
            for j in range(n):
                p, q = node(i-1, j), node(i, j)
                bars.append((p, q))
                k_e.append(self.kk_v[i, j])
                c_e.append(self.cc_v[i, j])
                kn_e.append(self.kn_v[i, j])

        return np.asarray(bars), np.asarray(k_e), np.asarray(c_e), np.asarray(kn_e)
        
    @staticmethod
    def _rotate_local(k_local, c_local, dx, dy):
        """
        Return rotated (4x4) axial stiffness & damping blocks
        given the current bar direction (dx,dy).
        """
        L = np.hypot(dx, dy)
        if L < grid_corotational.EPS_len:
            L = grid_corotational.EPS_len
        c = dx / L
        s = dy / L
        T = np.array([[ c, s, 0, 0],
                    [-s, c, 0, 0],
                    [ 0, 0, c, s],
                    [ 0, 0,-s, c]])
        k_rot = T.T @ k_local @ T
        c_rot = T.T @ c_local @ T
        return k_rot, c_rot          #  divide by L for truss

    # ---------- bar-by-bar assembly each call -------------------------------
    def assemble_KC(self, q, v):
        """
        Build global K(q) and C(q) [optionally sparse] every time step.
        q, v are (2N,) vectors of nodal displacements / velocities.
        """
        if self.sparse:
            K = scipy.sparse.lil_matrix((2*self.N, 2*self.N))
            C = scipy.sparse.lil_matrix((2*self.N, 2*self.N))
        else:
            K = np.zeros((2*self.N, 2*self.N))
            C = np.zeros_like(K)

        # loop over bars -----------------------------------------------------
        for e, (p, q_) in enumerate(self.bars):
            # global dof indices --------------------------------------------
            idx = np.array([2*p, 2*p+1, 2*q_, 2*q_+1])

            # current vector between the two nodes --------------------------
            dx = (self.node_coords[q_,0] + q[idx[2]]     # x_q
                  - self.node_coords[p,0] - q[idx[0]])
            dy = (self.node_coords[q_,1] + q[idx[3]]     # y_q
                  - self.node_coords[p,1] - q[idx[1]])

            k_blk, c_blk = self._rotate_local(self.k_local_scaled[e],
                                         self.c_local_scaled[e],
                                         dx, dy)
            # scatter --------------------------------------------------------
            for a in range(4):
                for b in range(4):
                    K[idx[a], idx[b]] += k_blk[a, b]
                    C[idx[a], idx[b]] += c_blk[a, b]
        # -------------------------------------------------------------
        #  add wall (grounded) springs & dash-pots if they are non-zero
        # -------------------------------------------------------------
        m, n = self.shape
        for i in range(m):
            p = 2 * (i * n + 0)          # node (i,0)  → left wall (x-DOF)
            k = self.kk_h[i, 0]
            c = self.cc_h[i, 0]
            if k != 0.0:
                K[p, p] += k
            if c != 0.0:
                C[p, p] += c

        for j in range(n):
            p = 2 * (0 * n + j) + 1      # node (0,j)  → top wall (y-DOF)
            k = self.kk_v[0, j]
            c = self.cc_v[0, j]
            if k != 0.0:
                K[p, p] += k
            if c != 0.0:
                C[p, p] += c

        return (K.tocsr(), C.tocsr()) if self.sparse else (K, C)
    
    def nonlin_transform(self, z):
        """
        Transform the state vector to compute nonlinear forces for co-rotational system.
        Computes nonlinear forces based on bar elongations and relative velocities.
        """
        if self.nonlinearity is not None:
            
            if len(z.shape) == 1:
                z = z.reshape(-1, 1)
            nt = z.shape[1]
            
            # Initialize arrays for bar elongations and rates
            bar_elongations = np.zeros((len(self.bars), nt))
            bar_rates = np.zeros((len(self.bars), nt))
            
            for t in range(nt):
                q_disp = z[:self.dofs, t]
                v_vel = z[self.dofs:, t]
                
                # Compute bar elongations and rates for each bar
                for e, (p_node_idx, q_node_idx) in enumerate(self.bars):
                    idx = np.array([2*p_node_idx, 2*p_node_idx+1, 2*q_node_idx, 2*q_node_idx+1])
                    
                    # Current bar vector
                    dx = (self.node_coords[q_node_idx,0] + q_disp[idx[2]]) - (self.node_coords[p_node_idx,0] + q_disp[idx[0]])
                    dy = (self.node_coords[q_node_idx,1] + q_disp[idx[3]]) - (self.node_coords[p_node_idx,1] + q_disp[idx[1]])
                    L = np.hypot(dx, dy)
                    if L < self.EPS_len:
                        L = self.EPS_len
                    
                    # Bar elongation (difference from rest length)
                    bar_elongations[e, t] = L - self.L0_e[e]
                    
                    # Bar elongation rate (projection of relative velocity onto bar axis)
                    if L > 0:
                        ex, ey = dx/L, dy/L
                        bar_rates[e, t] = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                                          ey*(v_vel[idx[3]]-v_vel[idx[1]]))
                    else:
                        bar_rates[e, t] = 0.0
            
            # Apply nonlinearity to bar elongations and rates
            nonlinear_stiffness_forces = self.nonlinearity.gk_func(bar_elongations, bar_rates)
            nonlinear_damping_forces = self.nonlinearity.gc_func(bar_elongations, bar_rates)
            
            # Convert bar forces back to nodal forces
            nodal_forces = np.zeros((self.dofs, nt))
            
            for t in range(nt):
                for e, (p_node_idx, q_node_idx) in enumerate(self.bars):
                    idx = np.array([2*p_node_idx, 2*p_node_idx+1, 2*q_node_idx, 2*q_node_idx+1])
                    
                    # Current bar direction
                    q_disp = z[:self.dofs, t]
                    dx = (self.node_coords[q_node_idx,0] + q_disp[idx[2]]) - (self.node_coords[p_node_idx,0] + q_disp[idx[0]])
                    dy = (self.node_coords[q_node_idx,1] + q_disp[idx[3]]) - (self.node_coords[p_node_idx,1] + q_disp[idx[1]])
                    L = np.hypot(dx, dy)
                    if L < self.EPS_len:
                        L = self.EPS_len
                    
                    if L > 0:
                        ex, ey = dx/L, dy/L
                        
                        # Nonlinear stiffness force
                        S_k = self.kn_e[e] * nonlinear_stiffness_forces[e, t]
                        # Nonlinear damping force  
                        S_c = self.kn_e[e] * nonlinear_damping_forces[e, t]
                        
                        # Total nonlinear force
                        S = S_k + S_c
                        S = np.clip(S, -self.S_MAX, self.S_MAX)
                        
                        # Distribute force to nodes
                        F_element = S * np.array([-ex, -ey, ex, ey])
                        nodal_forces[idx, t] += F_element
            
            # Return concatenated stiffness and damping forces
            return np.concatenate((
                nodal_forces,
                np.zeros_like(nodal_forces)  # No velocity-dependent forces in this implementation
            ))
        
        else:
            return np.zeros_like(z)
    
    def internal_force(self, q_disp, v_vel): # Renamed q, v for clarity within method
        """
        Assemble f_int(q,v) including bar forces, wall spring forces, and nonlinear forces.
        Returns f_int as a (dofs,) array.
        """
        f_int = np.zeros(self.dofs)
        # C_bar_v = np.zeros_like(f_int) # Not needed if f_int contains total damping

        coords = self.node_coords
        # --- Bar forces (elastic + damping) ---
        for e, (p_node_idx, q_node_idx) in enumerate(self.bars): # Corrected variable names
            idx = np.array([2*p_node_idx, 2*p_node_idx+1, 2*q_node_idx, 2*q_node_idx+1])

            # dx, dy are components of vector from node p to node q in current config
            dx = (coords[q_node_idx,0] + q_disp[idx[2]]) - (coords[p_node_idx,0] + q_disp[idx[0]])
            dy = (coords[q_node_idx,1] + q_disp[idx[3]]) - (coords[p_node_idx,1] + q_disp[idx[1]])
            L  = np.hypot(dx, dy)
            if L < self.EPS_len: # Use self.EPS_len consistently
                L = self.EPS_len
            L0 = self.L0_e[e]

            dL  = L - L0
            # Relative velocities projected onto the bar axis
            dLt = (dx*(v_vel[idx[2]]-v_vel[idx[0]]) +
                dy*(v_vel[idx[3]]-v_vel[idx[1]])) / L

            S  = self.k_e[e]*dL + self.c_e[e]*dLt
            S = np.clip(S, -self.S_MAX, self.S_MAX) # Use self.S_MAX

            ex, ey = dx/L, dy/L
            # Force vector in global DOFs for the element
            F_element = S * np.array([ -ex, -ey, ex, ey ]) # Standard: -ex for node p_node_idx_x, ex for q_node_idx_x
                                                            # Ensure your convention for F matches assembly.
                                                            # If F is force on nodes *by* element, then for node p it's S*(-dir_vec) and for node q it's S*(dir_vec)
                                                            # If dir_vec is (ex, ey) from p to q, then force on p is -S*(ex,ey) and force on q is S*(ex,ey).
                                                            # So element force contribution on [px,py, qx,qy] is S*[-ex, -ey, ex, ey]

            f_int[idx] += F_element # This was S * np.array([ex, ey, -ex, -ey]), check sign convention carefully.
                                    # If F = S * [ex, ey, -ex, -ey] means force AT p is (S*ex, S*ey) and AT q is (-S*ex, -S*ey)
                                    # This implies the vector from p to q is (-ex, -ey) for S to be positive in tension.
                                    # Standard derivation: internal force vector for [u_p_x, u_p_y, u_q_x, u_q_y] is S * [-c, -s, c, s] where c=ex, s=ey

        # --- Wall spring forces (elastic + damping) ---
        m_shape, n_shape = self.shape

        # Horizontal springs to left wall (node (i,0) x-dof)
        # These springs are defined by self.kk_h[i,0] and self.cc_h[i,0]
        for i in range(m_shape):
            node_idx_global = i * n_shape + 0 # Global index of node (i,0)
            dof_x = 2 * node_idx_global       # x-DOF for this node
            
            current_q_x = q_disp[dof_x]
            current_v_x = v_vel[dof_x]
            
            # Force = k*q + c*v. Internal force opposes displacement/velocity.
            # If q_x is positive (rightward displacement), spring pulls left (-k*q_x)
            f_int[dof_x] += (self.kk_h[i, 0] * current_q_x + self.cc_h[i, 0] * current_v_x)

        # Vertical springs to top wall (node (0,j) y-dof)
        # These springs are defined by self.kk_v[0,j] and self.cc_v[0,j]
        for j in range(n_shape):
            node_idx_global = 0 * n_shape + j # Global index of node (0,j)
            dof_y = 2 * node_idx_global + 1   # y-DOF for this node
            
            current_q_y = q_disp[dof_y]
            current_v_y = v_vel[dof_y]
            
            # If q_y is positive (downward displacement), spring pulls up (-k*q_y)
            f_int[dof_y] += (self.kk_v[0, j] * current_q_y + self.cc_v[0, j] * current_v_y)
        
        # Co-rotational boundary condition forces
        if 'nodes' in self.boundary_conditions and 'anchor_points' in self.boundary_conditions and 'springs' in self.boundary_conditions:
            for node, anchor_point, spring_props in zip(self.boundary_conditions['nodes'], 
                                                      self.boundary_conditions['anchor_points'],
                                                      self.boundary_conditions['springs']):
                if len(spring_props) >= 2:  # [k, c]
                    k, c = spring_props[0], spring_props[1]
                    
                    # Current position of the node
                    node_x = self.node_coords[node, 0] + q_disp[2*node]
                    node_y = self.node_coords[node, 1] + q_disp[2*node+1]
                    
                    # Vector from anchor to current node position
                    dx = node_x - anchor_point[0]
                    dy = node_y - anchor_point[1]
                    L = np.hypot(dx, dy)
                    
                    if L > self.EPS_len:
                        # Unit vector in direction of spring
                        ex, ey = dx/L, dy/L
                        
                        # Spring elongation (current length - rest length)
                        # Rest length is the initial distance from anchor to node
                        rest_length = np.hypot(self.node_coords[node, 0] - anchor_point[0],
                                             self.node_coords[node, 1] - anchor_point[1])
                        elongation = L - rest_length
                        
                        # Spring elongation rate (projection of velocity onto spring direction)
                        elongation_rate = ex * v_vel[2*node] + ey * v_vel[2*node+1]
                        
                        # Total spring force
                        spring_force = k * elongation + c * elongation_rate
                        
                        # Apply force in direction of spring (toward anchor point)
                        f_int[2*node] += spring_force * (-ex)  # Force opposes displacement
                        f_int[2*node+1] += spring_force * (-ey)

        # Note: Nonlinear forces are handled by nonlin_transform method
        # and applied by the simulator, so we don't add them here
                
        return f_int # Return only the total internal force vector

        
class arbitrary_truss_corotational(mdof_system):
    """
    Co-rotational truss system with randomly distributed nodes.
    Takes node coordinates, bar connectivity, and bar properties directly.
    Tracks relative displacements, velocities, and accelerations from initial positions.
    """
    
    EPS_len = 1e-9  # minimum bar length used in any 1/L
    S_MAX = 1e12  # optional force clipping

    def __init__(self,
                 node_coords, 
                 bar_connectivity, 
                 bar_masses, 
                 bar_stiffnesses, 
                 bar_dampings, 
                 bar_nonlinear_stiffnesses=None, 
                 boundary_conditions=None,
                 nonlinearity=None, sparse=True
                 ):
        """
        Initialize random truss system.
        
        Args:
            node_coords: (N, 2) array of node coordinates [x, y]
            bar_connectivity: (M, 2) array of bar connections [node1, node2]
            bar_masses: (M,) array of masses for each bar (distributed to nodes)
            bar_stiffnesses: (M,) array of linear stiffness for each bar
            bar_dampings: (M,) array of damping for each bar
            bar_nonlinear_stiffnesses: (M,) array of nonlinear stiffness for each bar (optional)
            boundary_conditions: dict with 'nodes', 'anchor_points', and 'springs' for grounded nodes (optional)
                                 'nodes': list of node indices
                                 'anchor_points': list of [x, y] coordinates for fixed anchor points
                                 'springs': list of [k, c] spring-damper properties [stiffness, damping]
            nonlinearity: nonlinearity object (optional)
            sparse: whether to use sparse matrices (default: True)
        """
        
        self.node_coords = np.array(node_coords)
        self.bar_connectivity = np.array(bar_connectivity)
        self.bar_masses = np.array(bar_masses)
        self.bar_stiffnesses = np.array(bar_stiffnesses)
        self.bar_dampings = np.array(bar_dampings)
        self.bar_nonlinear_stiffnesses = (np.array(bar_nonlinear_stiffnesses) 
                                         if bar_nonlinear_stiffnesses is not None 
                                         else np.zeros_like(bar_stiffnesses))
        self.boundary_conditions = boundary_conditions or {}
        self.nonlinearity = nonlinearity
        self.sparse = sparse
        
        # Number of nodes and bars
        self.nN = self.node_coords.shape[0]  # number of nodes
        self.nM = int(self.bar_connectivity.shape[0])  # number of bars
        self.dofs = 2 * self.nN  # 2 DOFs per node (x, y)
        
        # Validate inputs
        if self.bar_connectivity.shape[1] != 2:
            raise ValueError("bar_connectivity must be (M, 2) array")
        if len(self.bar_masses) != self.nM:
            raise ValueError("bar_masses length must match number of bars")
        if len(self.bar_stiffnesses) != self.nM:
            raise ValueError("bar_stiffnesses length must match number of bars")
        if len(self.bar_dampings) != self.nM:
            raise ValueError("bar_dampings length must match number of bars")
        if len(self.bar_nonlinear_stiffnesses) != self.nM:
            raise ValueError("bar_nonlinear_stiffnesses length must match number of bars")
        
        # Check for valid node indices
        if np.max(self.bar_connectivity) >= self.nN:
            raise ValueError("bar_connectivity contains invalid node indices")
        if np.min(self.bar_connectivity) < 0:
            raise ValueError("bar_connectivity contains negative node indices")
        
        # Validate boundary conditions
        if self.boundary_conditions:
            if not all(key in self.boundary_conditions for key in ['nodes', 'anchor_points', 'springs']):
                raise ValueError("boundary_conditions must contain 'nodes', 'anchor_points', and 'springs'")
            if len(self.boundary_conditions['nodes']) != len(self.boundary_conditions['anchor_points']):
                raise ValueError("Number of nodes and anchor points must match")
            if len(self.boundary_conditions['nodes']) != len(self.boundary_conditions['springs']):
                raise ValueError("Number of nodes and springs must match")
        
        # Build mass matrix by distributing bar masses to nodes
        self._build_mass_matrix()
        
        # Generate rest lengths from initial bar coordinates
        self._generate_rest_lengths()
        
        # Initialize base class with zero matrices (they're not used in co-rotational)
        Z = np.zeros((self.dofs, self.dofs))
        super().__init__(self.M_matrix, Z, Z, nonlinearity=nonlinearity)
        
        # Local axial bar matrices
        self._k_axial = np.array([[ 1, 0, -1, 0],
                                 [ 0, 0,  0, 0],
                                 [-1, 0,  1, 0],
                                 [ 0, 0,  0, 0]], dtype=float)
        self._c_axial = self._k_axial.copy()
        
        # Scaled local matrices
        self.k_local_scaled = (self._k_axial[None,:,:] * self.bar_stiffnesses[:,None,None])
        self.c_local_scaled = (self._c_axial[None,:,:] * self.bar_dampings[:,None,None])
    
    def _build_mass_matrix(self):
        """Build mass matrix by distributing bar masses to connected nodes."""
        self.M_matrix = np.zeros((self.dofs, self.dofs))
        
        # Distribute each bar's mass equally to its two nodes
        for i, (node1, node2) in enumerate(self.bar_connectivity):
            mass_per_node = self.bar_masses[i] / 2.0
            
            # Add mass to node1 (x and y DOFs)
            self.M_matrix[2*node1, 2*node1] += mass_per_node
            self.M_matrix[2*node1+1, 2*node1+1] += mass_per_node
            
            # Add mass to node2 (x and y DOFs)
            self.M_matrix[2*node2, 2*node2] += mass_per_node
            self.M_matrix[2*node2+1, 2*node2+1] += mass_per_node
    
    def _generate_rest_lengths(self):
        """Generate rest lengths for all bars from initial coordinates."""
        self.rest_lengths = np.zeros(self.nM)
        
        for i, (node1, node2) in enumerate(self.bar_connectivity):
            dx = self.node_coords[node2, 0] - self.node_coords[node1, 0]
            dy = self.node_coords[node2, 1] - self.node_coords[node1, 1]
            self.rest_lengths[i] = np.hypot(dx, dy)
    
    @staticmethod
    def _rotate_local(k_local, c_local, dx, dy):
        """
        Return rotated (4x4) axial stiffness & damping blocks
        given the current bar direction (dx,dy).
        """
        L = np.hypot(dx, dy)
        if L < arbitrary_truss_corotational.EPS_len:
            L = arbitrary_truss_corotational.EPS_len
        c = dx / L
        s = dy / L
        T = np.array([[ c, s, 0, 0],
                     [-s, c, 0, 0],
                     [ 0, 0, c, s],
                     [ 0, 0,-s, c]])
        k_rot = T.T @ k_local @ T
        c_rot = T.T @ c_local @ T
        return k_rot, c_rot
    
    def assemble_KC(self, q, v):
        """
        Build global K(q) and C(q) [optionally sparse] every time step.
        q, v are (2N,) vectors of nodal displacements / velocities.
        """
        if self.sparse:
            K = scipy.sparse.lil_matrix((self.dofs, self.dofs))
            C = scipy.sparse.lil_matrix((self.dofs, self.dofs))
        else:
            K = np.zeros((self.dofs, self.dofs))
            C = np.zeros_like(K)

        # Loop over bars
        for e, (node1, node2) in enumerate(self.bar_connectivity):
            # Global DOF indices
            idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])

            # Current vector between the two nodes
            dx = (self.node_coords[node2, 0] + q[idx[2]] - 
                  self.node_coords[node1, 0] - q[idx[0]])
            dy = (self.node_coords[node2, 1] + q[idx[3]] - 
                  self.node_coords[node1, 1] - q[idx[1]])

            k_blk, c_blk = self._rotate_local(self.k_local_scaled[e],
                                            self.c_local_scaled[e],
                                            dx, dy)
            
            # Scatter to global matrices
            for a in range(4):
                for b in range(4):
                    K[idx[a], idx[b]] += k_blk[a, b]
                    C[idx[a], idx[b]] += c_blk[a, b]
        
        # Add co-rotational boundary condition springs
        if 'nodes' in self.boundary_conditions and 'anchor_points' in self.boundary_conditions and 'springs' in self.boundary_conditions:
            for node, anchor_point, spring_props in zip(self.boundary_conditions['nodes'], 
                                                      self.boundary_conditions['anchor_points'],
                                                      self.boundary_conditions['springs']):
                if len(spring_props) >= 2:  # [k, c]
                    k, c = spring_props[0], spring_props[1]
                    
                    # Current position of the node
                    node_x = self.node_coords[node, 0] + q[2*node]
                    node_y = self.node_coords[node, 1] + q[2*node+1]
                    
                    # Vector from anchor to current node position
                    dx = node_x - anchor_point[0]
                    dy = node_y - anchor_point[1]
                    L = np.hypot(dx, dy)
                    
                    if L > self.EPS_len:
                        # Unit vector in direction of spring
                        ex, ey = dx/L, dy/L
                        
                        # Co-rotational spring stiffness matrix (2x2 for single node)
                        # This is the derivative of the spring force with respect to node position
                        k_spring = k * np.array([[ex*ex, ex*ey],
                                               [ey*ex, ey*ey]])
                        
                        # Co-rotational damping matrix
                        c_spring = c * np.array([[ex*ex, ex*ey],
                                               [ey*ex, ey*ey]])
                        
                        # Add to global matrices
                        node_idx = 2*node
                        K[node_idx:node_idx+2, node_idx:node_idx+2] += k_spring
                        C[node_idx:node_idx+2, node_idx:node_idx+2] += c_spring

        return (K.tocsr(), C.tocsr()) if self.sparse else (K, C)
    
    def nonlin_transform(self, z):
        """
        Transform the state vector to compute nonlinear forces for co-rotational system.
        Computes nonlinear forces based on bar elongations and relative velocities.
        """
        if self.nonlinearity is not None:
            
            if len(z.shape) == 1:
                z = z.reshape(-1, 1)
            nt = z.shape[1]
            
            # Initialize arrays for bar elongations and rates
            bar_elongations = np.zeros((self.nM, nt))
            bar_rates = np.zeros((self.nM, nt))
            
            for t in range(nt):
                q_disp = z[:self.dofs, t]
                v_vel = z[self.dofs:, t]
                
                # Compute bar elongations and rates for each bar
                for e, (node1, node2) in enumerate(self.bar_connectivity):
                    idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])
                    
                    # Current bar vector
                    dx = (self.node_coords[node2, 0] + q_disp[idx[2]] - 
                          self.node_coords[node1, 0] - q_disp[idx[0]])
                    dy = (self.node_coords[node2, 1] + q_disp[idx[3]] - 
                          self.node_coords[node1, 1] - q_disp[idx[1]])
                    L = np.hypot(dx, dy)
                    if L < self.EPS_len:
                        L = self.EPS_len
                    
                    # Bar elongation (difference from rest length)
                    bar_elongations[e, t] = L - self.rest_lengths[e]
                    
                    # Bar elongation rate (projection of relative velocity onto bar axis)
                    if L > 0:
                        ex, ey = dx/L, dy/L
                        bar_rates[e, t] = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                                          ey*(v_vel[idx[3]]-v_vel[idx[1]]))
                    else:
                        bar_rates[e, t] = 0.0
            
            # Apply nonlinearity to bar elongations and rates
            nonlinear_stiffness_forces = self.nonlinearity.gk_func(bar_elongations, bar_rates)
            nonlinear_damping_forces = self.nonlinearity.gc_func(bar_elongations, bar_rates)
            
            # Convert bar forces back to nodal forces
            nodal_forces = np.zeros((self.dofs, nt))
            
            for t in range(nt):
                for e, (node1, node2) in enumerate(self.bar_connectivity):
                    idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])
                    
                    # Current bar direction
                    q_disp = z[:self.dofs, t]
                    dx = (self.node_coords[node2, 0] + q_disp[idx[2]]) - (self.node_coords[node1, 0] + q_disp[idx[0]])
                    dy = (self.node_coords[node2, 1] + q_disp[idx[3]]) - (self.node_coords[node1, 1] + q_disp[idx[1]])
                    L = np.hypot(dx, dy)
                    if L < self.EPS_len:
                        L = self.EPS_len
                    
                    if L > 0:
                        ex, ey = dx/L, dy/L
                        
                        # Nonlinear stiffness force
                        S_k = self.bar_nonlinear_stiffnesses[e] * nonlinear_stiffness_forces[e, t]
                        # Nonlinear damping force  
                        S_c = self.bar_nonlinear_stiffnesses[e] * nonlinear_damping_forces[e, t]
                        
                        # Total nonlinear force
                        S = S_k + S_c
                        S = np.clip(S, -self.S_MAX, self.S_MAX)
                        
                        # Distribute force to nodes
                        F_element = S * np.array([-ex, -ey, ex, ey])
                        # F_element = S * np.array([ex, ey, -ex, -ey])
                        nodal_forces[idx, t] += F_element
            
            # Return concatenated stiffness and damping forces
            return np.concatenate((
                nodal_forces,
                np.zeros_like(nodal_forces)  # No velocity-dependent forces in this implementation
            ))
        
        else:
            return np.zeros_like(z)
    
    def internal_force(self, q_disp, v_vel):
        """
        Assemble f_int(q,v) including bar forces, boundary condition forces, and nonlinear forces.
        Returns f_int as a (dofs,) array.
        """
        if q_disp.ndim == 1:
            f_int = np.zeros(self.dofs)
        else:
            f_int = np.zeros((self.dofs, q_disp.shape[1]))

        # Bar forces (elastic + damping)
        for e, (node1, node2) in enumerate(self.bar_connectivity):
            idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])

            # Current bar vector
            dx = (self.node_coords[node2, 0] + q_disp[idx[2]] - 
                  self.node_coords[node1, 0] - q_disp[idx[0]])
            dy = (self.node_coords[node2, 1] + q_disp[idx[3]] - 
                  self.node_coords[node1, 1] - q_disp[idx[1]])
            L = np.hypot(dx, dy)
            if q_disp.ndim == 1:
                if L < self.EPS_len:
                    L = self.EPS_len
            else:
                L[L < self.EPS_len] = self.EPS_len
            L0 = self.rest_lengths[e]

            # Bar elongation and rate
            dL = L - L0
            if q_disp.ndim == 1:
                if L > 0:
                    ex, ey = dx/L, dy/L
                    dLt = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                           ey*(v_vel[idx[3]]-v_vel[idx[1]]))
                else:
                    dLt = 0.0
            else:
                dLt = np.zeros(L.shape)
                ex, ey = dx/L, dy/L
                dLt[L > 0] = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                              ey*(v_vel[idx[3]]-v_vel[idx[1]]))

            # Linear force
            S = self.bar_stiffnesses[e] * dL + self.bar_dampings[e] * dLt
            S = np.clip(S, -self.S_MAX, self.S_MAX)

            # Distribute force to nodes
            if q_disp.ndim == 1:
                if L > 0:
                    F_element = S * np.array([-ex, -ey, ex, ey])
                    # F_element = S * np.array([ex, ey, -ex, -ey])
                    # Debug print for all bars
                    # print(f"Bar {e}: dL={dL:.3e}, dLt={dLt:.3e}, S={S:.3e}, F_element={F_element}")
                    f_int[idx] += F_element
            else:
                F_element = S * np.array([-ex, -ey, ex, ey])
                # F_element = S * np.array([ex, ey, -ex, -ey])
                # Debug print for all bars, first time step
                # if F_element.shape[-1] > 0:
                    # print(f"Bar {e}: dL={dL[0]:.3e}, dLt={dLt[0]:.3e}, S={S[0]:.3e}, F_element={F_element[:,0]:.3e}")
                f_int[idx] += F_element

        # Boundary condition forces
        if 'nodes' in self.boundary_conditions and 'anchor_points' in self.boundary_conditions and 'springs' in self.boundary_conditions:
            for node, anchor_point, spring_props in zip(self.boundary_conditions['nodes'], 
                                                      self.boundary_conditions['anchor_points'],
                                                      self.boundary_conditions['springs']):
                if len(spring_props) >= 2:  # [k, c]
                    k, c = spring_props[0], spring_props[1]
                    if q_disp.ndim == 1:
                                                
                        # Co-rotational spring and damper
                        node_x = self.node_coords[node, 0] + q_disp[2*node]
                        node_y = self.node_coords[node, 1] + q_disp[2*node+1]
                        dx = node_x - anchor_point[0]
                        dy = node_y - anchor_point[1]
                        L = np.hypot(dx, dy)
                        if L > self.EPS_len:
                            ex, ey = dx/L, dy/L
                            rest_length = np.hypot(self.node_coords[node, 0] - anchor_point[0],
                                                   self.node_coords[node, 1] - anchor_point[1])
                            elongation = L - rest_length
                            elongation_rate = ex * v_vel[2*node] + ey * v_vel[2*node+1]
                            spring_force = k * elongation + c * elongation_rate
                            # Debug print for boundary spring
                            # print(f"Boundary spring: node={node}, elongation={elongation:.3e}, elongation_rate={elongation_rate:.3e}, spring_force={spring_force:.3e}, ex={ex:.3e}, ey={ey:.3e}")
                            f_int[2*node] += spring_force * (ex)
                            f_int[2*node+1] += spring_force * (ey)
                    else:
                        
                        # Co-rotational spring and damper
                        # q_disp, v_vel are (dofs, nt)
                        node_x = self.node_coords[node, 0] + q_disp[2*node, :]
                        node_y = self.node_coords[node, 1] + q_disp[2*node+1, :]
                        dx = node_x - anchor_point[0]
                        dy = node_y - anchor_point[1]
                        L = np.hypot(dx, dy)
                        mask = L > self.EPS_len
                        ex = np.zeros_like(L)
                        ey = np.zeros_like(L)
                        ex[mask] = dx[mask]/L[mask]
                        ey[mask] = dy[mask]/L[mask]
                        rest_length = np.hypot(self.node_coords[node, 0] - anchor_point[0],
                                               self.node_coords[node, 1] - anchor_point[1])
                        elongation = L - rest_length
                        elongation_rate = ex * v_vel[2*node, :] + ey * v_vel[2*node+1, :]
                        spring_force = k * elongation + c * elongation_rate
                        # Debug print for boundary spring (first time step)
                        # print(f"Boundary spring: node={node}, elongation={elongation[0]:.3e}, elongation_rate={elongation_rate[0]:.3e}, spring_force={spring_force[0]:.3e}, ex={ex[0]:.3e}, ey={ey[0]:.3e}")
                        f_int[2*node, :] += spring_force * (ex)
                        f_int[2*node+1, :] += spring_force * (ey)
                        
        return f_int
    
    def internal_forces(self, q_disp, v_vel):
        """
        Assemble f_int(q,v) including bar forces, boundary condition forces, and nonlinear forces.
        Returns f_int as a (dofs,) array.
        """
        if q_disp.ndim == 1:
            f_int = np.zeros(self.dofs)
            k_int = np.zeros(self.dofs)
            c_int = np.zeros(self.dofs)
        else:
            f_int = np.zeros((self.dofs, q_disp.shape[1]))
            k_int = np.zeros((self.dofs, q_disp.shape[1]))
            c_int = np.zeros((self.dofs, q_disp.shape[1]))

        # Bar forces (elastic + damping)
        for e, (node1, node2) in enumerate(self.bar_connectivity):
            idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])

            # Current bar vector
            dx = (self.node_coords[node2, 0] + q_disp[idx[2]] - 
                  self.node_coords[node1, 0] - q_disp[idx[0]])
            dy = (self.node_coords[node2, 1] + q_disp[idx[3]] - 
                  self.node_coords[node1, 1] - q_disp[idx[1]])
            L = np.hypot(dx, dy)
            if q_disp.ndim == 1:
                if L < self.EPS_len:
                    L = self.EPS_len
            else:
                L[L < self.EPS_len] = self.EPS_len
            L0 = self.rest_lengths[e]

            # Bar elongation and rate
            dL = L - L0
            if q_disp.ndim == 1:
                if L > 0:
                    ex, ey = dx/L, dy/L
                    dLt = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                           ey*(v_vel[idx[3]]-v_vel[idx[1]]))
                else:
                    dLt = 0.0
            else:
                dLt = np.zeros(L.shape)
                ex, ey = dx/L, dy/L
                dLt[L > 0] = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                              ey*(v_vel[idx[3]]-v_vel[idx[1]]))

            # Linear force
            S = self.bar_stiffnesses[e] * dL + self.bar_dampings[e] * dLt
            S = np.clip(S, -self.S_MAX, self.S_MAX)
            kS = self.bar_stiffnesses[e] * dL
            cS = self.bar_dampings[e] * dLt

            # Distribute force to nodes
            if q_disp.ndim == 1:
                if L > 0:
                    F_element = S * np.array([-ex, -ey, ex, ey])
                    # F_element = S * np.array([ex, ey, -ex, -ey])
                    # Debug print for all bars
                    # print(f"Bar {e}: dL={dL:.3e}, dLt={dLt:.3e}, S={S:.3e}, F_element={F_element}")
                    f_int[idx] += F_element
                    k_int[idx] += kS * np.array([-ex, -ey, ex, ey])
                    c_int[idx] += cS * np.array([-ex, -ey, ex, ey])
            else:
                F_element = S * np.array([-ex, -ey, ex, ey])
                # F_element = S * np.array([ex, ey, -ex, -ey])
                # Debug print for all bars, first time step
                # if F_element.shape[-1] > 0:
                    # print(f"Bar {e}: dL={dL[0]:.3e}, dLt={dLt[0]:.3e}, S={S[0]:.3e}, F_element={F_element[:,0]:.3e}")
                f_int[idx] += F_element
                k_int[idx] += kS * np.array([-ex, -ey, ex, ey])
                c_int[idx] += cS * np.array([-ex, -ey, ex, ey])

        # Boundary condition forces
        if 'nodes' in self.boundary_conditions and 'anchor_points' in self.boundary_conditions and 'springs' in self.boundary_conditions:
            for node, anchor_point, spring_props in zip(self.boundary_conditions['nodes'], 
                                                      self.boundary_conditions['anchor_points'],
                                                      self.boundary_conditions['springs']):
                if len(spring_props) >= 2:  # [k, c]
                    k, c = spring_props[0], spring_props[1]
                    if q_disp.ndim == 1:
                                                
                        # Co-rotational spring and damper
                        node_x = self.node_coords[node, 0] + q_disp[2*node]
                        node_y = self.node_coords[node, 1] + q_disp[2*node+1]
                        dx = node_x - anchor_point[0]
                        dy = node_y - anchor_point[1]
                        L = np.hypot(dx, dy)
                        if L > self.EPS_len:
                            ex, ey = dx/L, dy/L
                            rest_length = np.hypot(self.node_coords[node, 0] - anchor_point[0],
                                                   self.node_coords[node, 1] - anchor_point[1])
                            elongation = L - rest_length
                            elongation_rate = ex * v_vel[2*node] + ey * v_vel[2*node+1]
                            spring_force = k * elongation + c * elongation_rate
                            # Debug print for boundary spring
                            # print(f"Boundary spring: node={node}, elongation={elongation:.3e}, elongation_rate={elongation_rate:.3e}, spring_force={spring_force:.3e}, ex={ex:.3e}, ey={ey:.3e}")
                            f_int[2*node] += spring_force * (ex)
                            f_int[2*node+1] += spring_force * (ey)
                            k_int[2*node] += k * elongation * ex
                            k_int[2*node+1] += k * elongation * ey
                            c_int[2*node] += c * elongation_rate * ex
                            c_int[2*node+1] += c * elongation_rate * ey
                    else:
                        
                        # Co-rotational spring and damper
                        # q_disp, v_vel are (dofs, nt)
                        node_x = self.node_coords[node, 0] + q_disp[2*node, :]
                        node_y = self.node_coords[node, 1] + q_disp[2*node+1, :]
                        dx = node_x - anchor_point[0]
                        dy = node_y - anchor_point[1]
                        L = np.hypot(dx, dy)
                        mask = L > self.EPS_len
                        ex = np.zeros_like(L)
                        ey = np.zeros_like(L)
                        ex[mask] = dx[mask]/L[mask]
                        ey[mask] = dy[mask]/L[mask]
                        rest_length = np.hypot(self.node_coords[node, 0] - anchor_point[0],
                                               self.node_coords[node, 1] - anchor_point[1])
                        elongation = L - rest_length
                        elongation_rate = ex * v_vel[2*node, :] + ey * v_vel[2*node+1, :]
                        spring_force = k * elongation + c * elongation_rate
                        # Debug print for boundary spring (first time step)
                        # print(f"Boundary spring: node={node}, elongation={elongation[0]:.3e}, elongation_rate={elongation_rate[0]:.3e}, spring_force={spring_force[0]:.3e}, ex={ex[0]:.3e}, ey={ey[0]:.3e}")
                        f_int[2*node, :] += spring_force * (ex)
                        f_int[2*node+1, :] += spring_force * (ey)
                        k_int[2*node, :] += k * elongation * ex
                        k_int[2*node+1, :] += k * elongation * ey
                        c_int[2*node, :] += c * elongation_rate * ex
                        c_int[2*node+1, :] += c * elongation_rate * ey

        # Note: Nonlinear forces are handled by nonlin_transform method
        # and applied by the simulator, so we don't add them here
                
        return f_int, k_int, c_int
    
    def get_relative_kinematics(self, q_disp, v_vel, a_acc=None):
        """
        Get relative kinematics (displacement, velocity, acceleration) from initial positions.
        
        Args:
            q_disp: (2N,) displacement vector
            v_vel: (2N,) velocity vector
            a_acc: (2N,) acceleration vector (optional)
            
        Returns:
            dict with 'positions', 'velocities', 'accelerations' (if provided)
        """
        # Current positions = initial positions + displacements
        current_positions = self.node_coords + q_disp.reshape(-1, 2)
        
        # Velocities are already relative (no initial velocity)
        current_velocities = v_vel.reshape(-1, 2)
        
        result = {
            'positions': current_positions,
            'velocities': current_velocities
        }
        
        if a_acc is not None:
            current_accelerations = a_acc.reshape(-1, 2)
            result['accelerations'] = current_accelerations
            
        return result
    
    def get_bar_kinematics(self, q_disp, v_vel, a_acc=None):
        """
        Get bar-level kinematics (elongations, rates, accelerations).
        
        Args:
            q_disp: (2N,) displacement vector
            v_vel: (2N,) velocity vector
            a_acc: (2N,) acceleration vector (optional)
            
        Returns:
            dict with 'elongations', 'rates', 'accelerations' (if provided)
        """
        elongations = np.zeros(int(self.nM))
        rates = np.zeros(int(self.nM))
        
        for e, (node1, node2) in enumerate(self.bar_connectivity):
            idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])
            
            # Current bar vector
            dx = (self.node_coords[node2, 0] + q_disp[idx[2]] - 
                  self.node_coords[node1, 0] - q_disp[idx[0]])
            dy = (self.node_coords[node2, 1] + q_disp[idx[3]] - 
                  self.node_coords[node1, 1] - q_disp[idx[1]])
            L = np.hypot(dx, dy)
            if L < self.EPS_len:
                L = self.EPS_len
            
            # Elongation
            elongations[e] = L - self.rest_lengths[e]
            
            # Rate
            if L > 0:
                ex, ey = dx/L, dy/L
                rates[e] = (ex*(v_vel[idx[2]]-v_vel[idx[0]]) + 
                           ey*(v_vel[idx[3]]-v_vel[idx[1]]))
        
        result = {
            'elongations': elongations,
            'rates': rates
        }
        
        if a_acc is not None:
            accelerations = np.zeros(int(self.nM))
            for e, (node1, node2) in enumerate(self.bar_connectivity):
                idx = np.array([2*node1, 2*node1+1, 2*node2, 2*node2+1])
                
                # Current bar vector
                dx = (self.node_coords[node2, 0] + q_disp[idx[2]] - 
                      self.node_coords[node1, 0] - q_disp[idx[0]])
                dy = (self.node_coords[node2, 1] + q_disp[idx[3]] - 
                      self.node_coords[node1, 1] - q_disp[idx[1]])
                L = np.hypot(dx, dy)
                if L < self.EPS_len:
                    L = self.EPS_len
                
                # Acceleration
                if L > 0:
                    ex, ey = dx/L, dy/L
                    accelerations[e] = (ex*(a_acc[idx[2]]-a_acc[idx[0]]) + 
                                       ey*(a_acc[idx[3]]-a_acc[idx[1]]))
            
            result['accelerations'] = accelerations
            
        return result
    
    def get_internal_forces(self, z):
        """
        Returns a tuple (linear_forces, nonlinear_forces) for the current state z.
        Both are arrays of shape (dofs,) or (dofs, nt) depending on z.
        """
        if z.ndim == 1:
            q_disp = z[:self.dofs]
            v_vel = z[self.dofs:]
        else:
            q_disp = z[:self.dofs, :]
            v_vel = z[self.dofs:, :]
        linear_forces, k_int, c_int = self.internal_forces(q_disp, v_vel)
        nonlinear_forces_full = self.nonlin_transform(z)
        nonlinear_forces = nonlinear_forces_full[:self.dofs] if nonlinear_forces_full.shape[0] >= self.dofs else nonlinear_forces_full
        return linear_forces, k_int, c_int, nonlinear_forces
    
    def _energy_balances(self, z, external_forces=None, delta_time=None):
        """
        Compute energy balance residual for a trajectory z (2*dofs x nt):
        Returns energy residual matching the PyTorch implementation:
        energy_residual = mech_energy_diff - external_work - nonlinear_work + damping_work
        
        Args:
            z: state vector (2*dofs, nt)
            external_forces: external forces (dofs, nt) or None
            delta_time: time step or None (will try to infer)
            
        Returns:
            dict with energy components and residual (length nt-1 for residual)
        """
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        nt = z.shape[1]
        dofs = self.dofs
        
        # Extract displacements and velocities
        q = z[:dofs, :]  # displacements
        v = z[dofs:, :]  # velocities
        
        # Determine time step
        if delta_time is None:
            if hasattr(self, 't') and len(self.t) == nt:
                delta_time = self.t[1] - self.t[0]
            else:
                delta_time = 1.0
        
        # Kinetic energy per node: 0.5 * m * (vx^2 + vy^2)
        # Reshape mass matrix diagonal to (num_nodes,) for x,y pairs
        node_masses = self.M_matrix.diagonal()[::2]  # Take every other element (x DOFs)
        v_reshaped = v.reshape(self.nN, 2, nt)  # (num_nodes, 2, nt)
        kinetic_energy_per_node = 0.5 * node_masses[:, None] * np.sum(v_reshaped**2, axis=1)  # (num_nodes, nt)
        
        # Total kinetic energy over time
        KE_total = np.sum(kinetic_energy_per_node, axis=0)  # (nt,)
        delta_kinetic_energy = np.diff(KE_total)  # (nt-1,)
        
        # Bar potential energy and extensions
        bar_extensions = np.zeros((self.nM, nt))
        bar_rates = np.zeros((self.nM, nt))
        PE_bar_per_edge = np.zeros((self.nM, nt))
        
        for e, (node1, node2) in enumerate(self.bar_connectivity):
            # Current bar vector
            dx = (self.node_coords[node2, 0] + q[2*node2, :] - 
                  self.node_coords[node1, 0] - q[2*node1, :])
            dy = (self.node_coords[node2, 1] + q[2*node2+1, :] - 
                  self.node_coords[node1, 1] - q[2*node1+1, :])
            L = np.sqrt(dx**2 + dy**2)
            L0 = self.rest_lengths[e]
            
            # Bar extension (elongation)
            bar_extensions[e, :] = L - L0
            
            # Bar extension rate
            with np.errstate(divide='ignore', invalid='ignore'):
                ex = np.where(L > 0, dx / L, 0)
                ey = np.where(L > 0, dy / L, 0)
                bar_rates[e, :] = (ex * (v[2*node2, :] - v[2*node1, :]) + 
                                  ey * (v[2*node2+1, :] - v[2*node1+1, :]))
            
            # Potential energy for this bar
            PE_bar_per_edge[e, :] = 0.5 * self.bar_stiffnesses[e] * bar_extensions[e, :]**2
        
        # Total potential energy
        PE_bar = np.sum(PE_bar_per_edge, axis=0)  # (nt,)
        delta_potential_energy = np.diff(PE_bar)  # (nt-1,)
        
        # Mechanical energy change
        mech_energy_diff = delta_kinetic_energy + delta_potential_energy  # (nt-1,)
        
        # Damping power and work
        damping_power_per_edge = self.bar_dampings[:, None] * bar_rates**2  # (num_edges, nt)
        damping_power = np.sum(damping_power_per_edge, axis=0)  # (nt,)
        # Trapezoidal integration for damping work
        damping_work = 0.5 * (damping_power[1:] + damping_power[:-1]) * delta_time  # (nt-1,)
        
        # External work (if external forces provided)
        external_work = np.zeros(nt-1)
        if external_forces is not None:
            # external_forces should be (dofs, nt)
            if external_forces.shape != (dofs, nt):
                raise ValueError(f"external_forces shape {external_forces.shape} != expected {(dofs, nt)}")
            
            # Power from external forces: F · v
            external_power_per_node = np.sum(external_forces.reshape(self.nN, 2, nt) * v_reshaped, axis=1)  # (num_nodes, nt)
            external_power = np.sum(external_power_per_node, axis=0)  # (nt,)
            # Trapezoidal integration for external work
            external_work = 0.5 * (external_power[1:] + external_power[:-1]) * delta_time  # (nt-1,)
        
        # Nonlinear work (if nonlinearity present)
        nonlinear_work = np.zeros(nt-1)
        if self.nonlinearity is not None:
            # Get nonlinear forces
            nonlinear_forces_full = self.nonlin_transform(z)  # (2*dofs, nt)
            nonlinear_forces = nonlinear_forces_full[:dofs, :]  # (dofs, nt)
            
            # Power from nonlinear forces: F_nl · v
            nonlinear_power_per_node = np.sum(nonlinear_forces.reshape(self.nN, 2, nt) * v_reshaped, axis=1)  # (num_nodes, nt)
            nonlinear_power = np.sum(nonlinear_power_per_node, axis=0)  # (nt,)
            # Trapezoidal integration for nonlinear work
            nonlinear_work = 0.5 * (nonlinear_power[1:] + nonlinear_power[:-1]) * delta_time  # (nt-1,)
        
        # Boundary condition potential energy
        PE_bc = np.zeros(nt)
        if 'nodes' in self.boundary_conditions and 'anchor_points' in self.boundary_conditions and 'springs' in self.boundary_conditions:
            for i, (node, springs) in enumerate(zip(self.boundary_conditions['nodes'], self.boundary_conditions['springs'])):
                anchor_x, anchor_y = self.boundary_conditions['anchor_points'][i]
                k = springs[0]  # Assuming same spring constant for both directions
                x = self.node_coords[node, 0] + q[2*node, :]
                y = self.node_coords[node, 1] + q[2*node+1, :]
                # Distance from anchor point
                dx_anchor = x - anchor_x
                dy_anchor = y - anchor_y
                L_anchor = np.sqrt(dx_anchor**2 + dy_anchor**2)
                rest_length_anchor = np.hypot(self.node_coords[node, 0] - anchor_x,
                                            self.node_coords[node, 1] - anchor_y)
                elongation_anchor = L_anchor - rest_length_anchor
                PE_bc += 0.5 * k * elongation_anchor**2
        
        # Energy residual (matches PyTorch implementation)
        energy_residual = mech_energy_diff - external_work - nonlinear_work + damping_work  # (nt-1,)
        
        # Total energies for reference
        total_energy = KE_total + PE_bar + PE_bc
        
        result = {
            'KE_total': KE_total,  # (nt,)
            'PE_total': PE_bar + PE_bc,  # (nt,)
            'KE_diff': delta_kinetic_energy,  # (nt-1,)
            'PE_diff': delta_potential_energy,  # (nt-1,)
            'total_energy': total_energy,  # (nt,)
            'damping_work': damping_work,  # (nt-1,)
            'external_work': external_work,  # (nt-1,)
            'nonlinear_work': nonlinear_work,  # (nt-1,)
            'mech_energy_diff': mech_energy_diff,  # (nt-1,)
            'energy_residual': energy_residual,  # (nt-1,)
            'delta_time': delta_time
        }
        
        return result, energy_residual
    
    def assess_energy_conservation(self, z, external_forces=None, delta_time=None, verbose=True):
        """
        Assess energy conservation quality by comparing residual to characteristic energy scales.
        
        Args:
            z: state vector (2*dofs, nt)
            external_forces: external forces (dofs, nt) or None
            delta_time: time step or None
            verbose: whether to print assessment
            
        Returns:
            dict with assessment metrics
        """
        result, energy_residual = self._energy_balances(z, external_forces, delta_time)
        
        # Characteristic energy scales
        KE_max = np.max(result['KE_total'])
        PE_max = np.max(result['PE_total'])
        total_energy_max = np.max(result['total_energy'])
        
        # Energy change scales
        KE_change_rms = np.sqrt(np.mean(result['KE_diff']**2)) if len(result['KE_diff']) > 0 else 0
        PE_change_rms = np.sqrt(np.mean(result['PE_diff']**2)) if len(result['PE_diff']) > 0 else 0
        
        # Work scales
        damping_work_rms = np.sqrt(np.mean(result['damping_work']**2)) if len(result['damping_work']) > 0 else 0
        external_work_rms = np.sqrt(np.mean(result['external_work']**2)) if len(result['external_work']) > 0 else 0
        nonlinear_work_rms = np.sqrt(np.mean(result['nonlinear_work']**2)) if len(result['nonlinear_work']) > 0 else 0
        
        # Residual statistics
        residual_rms = np.sqrt(np.mean(energy_residual**2))
        residual_max = np.max(np.abs(energy_residual))
        residual_mean = np.mean(energy_residual)
        
        # Relative errors (avoid division by zero)
        rel_error_total = residual_rms / (total_energy_max + 1e-12)
        rel_error_ke = residual_rms / (KE_max + 1e-12)
        rel_error_pe = residual_rms / (PE_max + 1e-12)
        
        # Assessment
        assessment = {
            'residual_rms': residual_rms,
            'residual_max': residual_max,
            'residual_mean': residual_mean,
            'KE_max': KE_max,
            'PE_max': PE_max,
            'total_energy_max': total_energy_max,
            'rel_error_total': rel_error_total,
            'rel_error_ke': rel_error_ke,
            'rel_error_pe': rel_error_pe,
            'KE_change_rms': KE_change_rms,
            'PE_change_rms': PE_change_rms,
            'damping_work_rms': damping_work_rms,
            'external_work_rms': external_work_rms,
            'nonlinear_work_rms': nonlinear_work_rms
        }
        
        if verbose:
            print("Energy Conservation Assessment:")
            print("=" * 50)
            print(f"Energy Residual RMS:     {residual_rms:.6f}")
            print(f"Energy Residual Max:     {residual_max:.6f}")
            print(f"Energy Residual Mean:    {residual_mean:.6f}")
            print()
            print("Characteristic Energy Scales:")
            print(f"Max Kinetic Energy:      {KE_max:.6f}")
            print(f"Max Potential Energy:    {PE_max:.6f}")
            print(f"Max Total Energy:        {total_energy_max:.6f}")
            print()
            print("Relative Errors:")
            print(f"Relative to Total Energy: {rel_error_total:.2e} ({rel_error_total*100:.4f}%)")
            print(f"Relative to Max KE:       {rel_error_ke:.2e} ({rel_error_ke*100:.4f}%)")
            print(f"Relative to Max PE:       {rel_error_pe:.2e} ({rel_error_pe*100:.4f}%)")
            print()
            print("Energy Change Scales (RMS):")
            print(f"KE Changes:              {KE_change_rms:.6f}")
            print(f"PE Changes:              {PE_change_rms:.6f}")
            print(f"Damping Work:            {damping_work_rms:.6f}")
            print(f"External Work:           {external_work_rms:.6f}")
            print(f"Nonlinear Work:          {nonlinear_work_rms:.6f}")
            print()
            
            # Qualitative assessment
            if rel_error_total < 1e-6:
                quality = "EXCELLENT"
            elif rel_error_total < 1e-4:
                quality = "VERY GOOD"
            elif rel_error_total < 1e-2:
                quality = "GOOD"
            elif rel_error_total < 1e-1:
                quality = "ACCEPTABLE"
            else:
                quality = "POOR"
                
            print(f"Energy Conservation Quality: {quality}")
            print(f"(Relative error: {rel_error_total:.2e})")
        
        return assessment
        
   
def find_nearest_neighbors(coords, k=3, min_connections=2):
    """Find k nearest neighbors for each node, ensuring minimum connectivity."""
    from scipy.spatial.distance import cdist
    
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)  # Exclude self
    
    connections = []
    node_connection_count = np.zeros(len(coords), dtype=int)
    
    # First pass: connect each node to its k nearest neighbors
    for i in range(len(coords)):
        nearest = np.argsort(distances[i])[:k]
        for j in nearest:
            if i < j:  # Avoid duplicate connections
                connections.append([i, j])
                node_connection_count[i] += 1
                node_connection_count[j] += 1
    
    # Second pass: ensure minimum connectivity
    for i in range(len(coords)):
        while node_connection_count[i] < min_connections:
            # Find the closest unconnected node
            available_nodes = []
            for j in range(len(coords)):
                if i != j and node_connection_count[j] < k:  # Don't exceed k connections
                    # Check if connection already exists
                    connection_exists = False
                    for conn in connections:
                        if (conn[0] == i and conn[1] == j) or (conn[0] == j and conn[1] == i):
                            connection_exists = True
                            break
                    
                    if not connection_exists:
                        available_nodes.append((j, distances[i, j]))
            
            if available_nodes:
                # Sort by distance and take the closest
                available_nodes.sort(key=lambda x: x[1])
                j = available_nodes[0][0]
                
                # Add connection
                connections.append([min(i, j), max(i, j)])
                node_connection_count[i] += 1
                node_connection_count[j] += 1
            else:
                # No available nodes, break to avoid infinite loop
                break
    
    return connections

def create_delaunay_connections(coords):
    """
    Generates truss connections using Delaunay triangulation for an even layout.

    Args:
        coords (np.ndarray): A NumPy array of shape (N, 2) or (N, 3) with node coordinates.

    Returns:
        list: A list of lists, where each inner list represents a connection [node1, node2].
    """
    if len(coords) < 3:
        # Delaunay needs at least 3 points in 2D or 4 in 3D.
        # Handle this edge case as needed, e.g., connect all to all.
        if len(coords) == 2:
            return [[0, 1]]
        return []

    # 1. Perform the Delaunay triangulation
    tri = Delaunay(coords)

    # 2. Extract unique edges from the triangles (simplices)
    edges = set()
    # Each simplex is a triangle defined by the indices of its 3 vertices
    for simplex in tri.simplices:
        # For each triangle, add its three edges to the set
        # Using a sorted tuple (i, j) with i < j ensures uniqueness
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[2], simplex[0]))))

    # 3. Convert the set of tuples back to a list of lists
    return [list(edge) for edge in edges]
        