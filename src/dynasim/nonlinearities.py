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
    
class truss_nonlinearity(nonlinearity):

    def __init__(self, bar_nonlinear_stiffnesses, stiff_exponent=3, damp_exponent=0.5):
        self.bar_nonlinear_stiffnesses = np.array(bar_nonlinear_stiffnesses)
        self.stiff_exponent = stiff_exponent
        self.damp_exponent = damp_exponent
        self.n_bars = len(bar_nonlinear_stiffnesses)
    
    def gk_func(self, elongations, rates, angles):
        if self.stiff_exponent == 0.0:
            return np.zeros_like(elongations)
        else:
            return np.sign(elongations) * np.abs(elongations)**self.stiff_exponent
    
    def gc_func(self, elongations, rates, angles):
        if self.damp_exponent == 0.0:
            return np.zeros_like(rates)
        else:
            return np.sign(rates) * np.abs(rates)**self.damp_exponent
        
class pdelta_truss_nonlinearity(nonlinearity):
    """
    Nonlinearity class for P-delta effects in truss-based frame models.
    """
    
    def __init__(self, node_coords, bar_connectivity, vertical_loads, stiff_exponent=1.0):
        """
        Initialize P-delta nonlinearity for truss-based frame.
        
        Args:
            node_coords: Node coordinates array (N, 2)
            bar_connectivity: Bar connectivity array (M, 2)
            vertical_loads: Dictionary mapping node indices to vertical loads
            stiff_exponent: Exponent for nonlinear relationship (default: 1.0)
        """
        self.node_coords = node_coords
        self.bar_connectivity = bar_connectivity
        self.vertical_loads = vertical_loads
        self.stiff_exponent = stiff_exponent
        self.n_nodes = len(node_coords)
        self.n_bars = len(bar_connectivity)
        
        # Total DOFs: 2 per node (x, y)
        dofs = 2 * self.n_nodes
        super().__init__(dofs)
        
        # Create bar nonlinear stiffness array
        self.bar_nonlinear_stiffnesses = np.zeros(self.n_bars)
        
        # Identify vertical bars (columns) - approximation based on orientation
        for i, (node1, node2) in enumerate(bar_connectivity):
            dx = node_coords[node2, 0] - node_coords[node1, 0]
            dy = node_coords[node2, 1] - node_coords[node1, 1]
            
            # If bar is primarily vertical (column)
            if abs(dy) > abs(dx):
                # Find if any of the nodes has a vertical load
                for node_idx in [node1, node2]:
                    if node_idx in vertical_loads:
                        # Apply P-delta effect proportional to the vertical load
                        self.bar_nonlinear_stiffnesses[i] = vertical_loads[node_idx]
    
    def gk_func(self, elongations, rates, angle_deviation):
        """
        Compute nonlinear stiffness contribution based on bar elongations.
        
        Args:
            elongations: Bar elongations array
            rates: Bar elongation rates array
            
        Returns:
            Nonlinear force contribution
        """
        if self.stiff_exponent == 0.0:
            return np.zeros_like(elongations)
        else:
            # For P-delta, we want the effect to be proportional to lateral displacement
            # This is approximated through the bar elongations
            return elongations  # Linear relationship for P-delta
    
    def gc_func(self, elongations, rates, angle_deviation):
        """
        Compute nonlinear damping contribution.
        P-delta typically doesn't affect damping directly.
        
        Args:
            elongations: Bar elongations array
            rates: Bar elongation rates array
            
        Returns:
            Zero vector (no nonlinear damping from P-delta)
        """
        return np.zeros_like(rates)
    
class expansion_joint_nonlinearity(nonlinearity):
    """
    Nonlinearity class for expansion joint effects in truss-based frame models.
    """
    
    def __init__(self, angles_gap_sizes, expansion_stiffnesses):
        """
        Initialize expansion joint nonlinearity for truss-based frame.
        
        Args:
            gap_size: Gap size
            expansion_stiffness: Expansion stiffness
        """
        self.angles_gap_sizes = angles_gap_sizes.reshape(-1, 1)
        self.bar_nonlinear_stiffnesses = expansion_stiffnesses.reshape(-1, 1)
        self.output_type = 'moment'
    
    def gk_func(self, elongations, rates, angle_deviation):
        """
        Compute nonlinear stiffness contribution based on bar elongations.
        
        Args:
            elongations: Bar elongations array
            rates: Bar elongation rates array
            
        Returns:
            Nonlinear force contribution
        """
        
        moments = np.zeros_like(angle_deviation)
        active_mask = np.abs(angle_deviation) >= self.angles_gap_sizes
        
        if np.any(active_mask):
            moments[active_mask] = self.bar_nonlinear_stiffnesses[active_mask] * (
                np.abs(angle_deviation[active_mask]) - self.angles_gap_sizes[active_mask]
                ) * np.sign(angle_deviation[active_mask])
        
        return moments
    
    def gc_func(self, elongations, rates, angle_deviation):
        """
        Compute nonlinear damping contribution.
        P-delta typically doesn't affect damping directly.
        
        Args:
            elongations: Bar elongations array
            rates: Bar elongation rates array
            
        Returns:
            Zero vector (no nonlinear damping from P-delta)
        """
        return np.zeros_like(rates)

