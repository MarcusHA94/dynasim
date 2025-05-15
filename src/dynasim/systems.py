import numpy as np
from math import pi
from dynasim.base import mdof_system, cont_ss_system
import scipy.integrate as integrate
import scipy

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

            x_ = z[:self.dofs] - np.concatenate((np.zeros_like(z[:1]), z[:self.dofs-1]))
            x_dot = z[self.dofs:] - np.concatenate((np.zeros_like(z[:1]), z[self.dofs:-1]))

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
            m, n       = mm.shape
        elif shape is not None:      # scalar → fill arrays
            m, n       = shape
            self.mm_   = mm   * np.ones(shape)
            self.cc_h  = cc_h * np.ones(shape)
            self.cc_v  = cc_v * np.ones(shape)
            self.kk_h  = kk_h * np.ones(shape)
            self.kk_v  = kk_v * np.ones(shape)
        else:
            raise ValueError("Either provide full arrays or the 'shape'.")

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
        self.bars, self.k_e, self.c_e = self._make_bar_arrays()
        
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

        # -------- vertical bars (between (i-1,j) ↔ (i,j)) ----------------
        for i in range(1, m):                            # skip top wall row
            for j in range(n):
                p, q = node(i-1, j), node(i, j)
                bars.append((p, q))
                k_e.append(self.kk_v[i, j])
                c_e.append(self.cc_v[i, j])

        return np.asarray(bars), np.asarray(k_e), np.asarray(c_e)
        
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
    
    def internal_force(self, q_disp, v_vel): # Renamed q, v for clarity within method
        """
        Assemble f_int(q,v) including bar forces and wall spring forces.
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
                
        return f_int # Return only the total internal force vector

        