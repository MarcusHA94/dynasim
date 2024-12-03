import numpy as np
import warnings

class nonlinearity:

    def __init__(self, dofs):

        self.dofs = dofs
        if self.dofs is not None:
            self.Cn = np.zeros((self.dofs, self.dofs))
            self.Kn = np.zeros((self.dofs, self.dofs))

    def gc_func(self, x, xdot):
        return np.zeros_like(xdot)
    
    def gk_func(self, x, xdot):
        return np.zeros_like(x)

class grid_exponent_stiffness(nonlinearity):
    
    def __init__(self, kn_h, kn_v, exponent=3, arr_shape=None):
        self.exponent = exponent
        match kn_h:
            case np.ndarray():
                self.kn_h = kn_h
                arr_shape = kn_h.shape
            case None:
                warnings.warn('No horizontal nonlinear stiffness parameters provided, proceeding with zero', UserWarning)
                self.kn_h = None
                arr_shape = None
            case _:
                if arr_shape is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.kn_h = None
                else:
                    self.kn_h = kn_h * np.ones(arr_shape)
        match kn_v:
            case np.ndarray():
                self.kn_v = kn_v
            case None:
                warnings.warn('No vertical nonlinear stiffness parameters provided, proceeding with zero', UserWarning)
                self.kn_v = None
            case _:
                if arr_shape is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.kn_v = None
                else:
                    self.kn_v = kn_v * np.ones(arr_shape)
        dofs = 2 * arr_shape[0] * arr_shape[1]
        m, n = arr_shape
        super().__init__(dofs)
        
        Kn = np.zeros((dofs, dofs))
        
        def index(i, j, direction):
            return 2 * (i * n + j) + direction # 0 for horizontal, 1 for vertical
        
        for i in range(m):
            for j in range(n):
                x_index = index(i, j, 0)
                y_index = index(i, j, 1)
                
                Kn[x_index, x_index] = self.kn_h[i, j]
                Kn[y_index, y_index] = self.kn_v[i, j]
                
                if (0 < j):
                    Kn[index(i, j-1, 0), x_index] = -self.kn_h[i, j]
                if (0 < i):
                    Kn[index(i-1, j, 1), y_index] = -self.kn_v[i, j]
                    
        self.Kn = Kn
    
    def gk_func(self, x, xdot):
        return np.sign(x) * np.abs(x)**self.exponent

class exponent_stiffness(nonlinearity):

    def __init__(self, kn_, exponent=3, dofs=None):
        self.exponent = exponent
        match kn_:
            case np.ndarray():
                self.kn_ = kn_
                dofs = kn_.shape[0]
            case None:
                warnings.warn('No nonlinear stiffness parameters provided, proceeding with zero', UserWarning)
                self.kn_ = None
                dofs = None
            case _:
                if dofs is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.kn_ = None
                else:
                    self.kn_ = kn_ * np.ones(dofs)
        super().__init__(dofs)

        self.Kn = np.diag(kn_) - np.diag(kn_[1:], 1)
        
    def gk_func(self, x, xdot):
        return np.sign(x) * np.abs(x)**self.exponent

class exponent_damping(nonlinearity):

    def __init__(self, cn_, exponent=0.5, dofs=None):
        self.exponent = exponent
        match cn_:
            case np.ndarray():
                self.cn_ = cn_
                dofs = cn_.shape[0]
            case None:
                warnings.warn('No nonlinear damping parameters provided, proceeding with zero', UserWarning)
                self.cn_ = None
                dofs = None
            case _:
                if dofs is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.cn_ = None
                else:
                    self.cn_ = cn_ * np.ones(dofs)
        super().__init__(dofs)
        
        self.Cn = np.diag(cn_) - np.diag(cn_[1:], 1)
    
    def gc_func(self, x, xdot):
        return np.sign(xdot) * np.abs(xdot)**self.exponent
    
class vanDerPol(nonlinearity):

    def __init__(self, cn_, dofs=None):
        match cn_:
            case np.ndarray():
                self.cn_ = cn_
                dofs = cn_.shape[0]
            case None:
                warnings.warn('No nonlinear damping parameters provided, proceeding with zero', UserWarning)
                self.cn_ = None
                dofs = None
            case _:
                if dofs is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.cn_ = None
                else:
                    self.cn_ = cn_ * np.ones(dofs)
        super().__init__(dofs)

        self.Cn = np.diag(cn_) - np.diag(cn_[1:], 1)

    def gc_func(self, x, xdot):
        return (x**2 - 1) * xdot
    
    

