import numpy as np
import scipy
import warnings

class simulator():

    def __init__(self, system):
        self.A = system.A
        self.H = system.H
        self.f = system.f
        self.t = system.t
        self.An = system.An
        self.dofs = system.dofs
        self.nonlin_transform = system.nonlin_transform
    
    def sim(self, time, z0):
        raise Exception('No simulator selected')
    
    def ode_f(self, t, z):
        match [self.An, self.f]:
            case [None,None]:
                return self.A@z
            case [_,None]:
                zn = self.nonlin_transform(z)
                return self.A@z + self.An@zn
            case [None,_]:
                t_id = np.argmin(np.abs(self.t-t))
                return self.A@z + self.H@self.f[:,t_id]
            case [_,_]:
                zn = self.nonlin_transform(z).squeeze()
                t_id = np.argmin(np.abs(self.t-t))
                return self.A@z + self.An@zn + self.H@self.f[:,t_id]
    

class scipy_ivp(simulator):

    def __init__(self, system):
        super().__init__(system)

    def sim(self, time, z0):

        tspan = (time[0], time[-1])
        results = scipy.integrate.solve_ivp(
            fun = self.ode_f,
            t_span = tspan,
            y0 = z0,
            t_eval = time
            )
        z = results.y
        return {
            'x' : np.array(z[:self.dofs,:]),
            'xdot' : np.array(z[self.dofs:,:])
        }


class rk4(simulator):

    def __init__(self, system):
        super().__init__(system)
        self.system = system

    def sim_one(self, dt, z0, t_point):

        k1 = self.ode_f(t_point, z0)
        k2 = self.ode_f(t_point, z0 + k1*dt/2)
        k3 = self.ode_f(t_point, z0 + k2*dt/2)
        k4 = self.ode_f(t_point, z0 + k3*dt)
        zplus1 = z0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return zplus1

    def sim(self, time, z0):

        self.ns = time.shape[0]
        dt = time[1]-time[0]
        z = np.zeros((2*self.system.dofs,self.ns))
        z[:,0] = z0
        for t in range(self.ns-1):
            k1 = self.ode_f(time[t], z[:,t])
            k2 = self.ode_f(time[t], z[:,t] + k1*dt/2)
            k3 = self.ode_f(time[t], z[:,t] + k2*dt/2)
            k4 = self.ode_f(time[t], z[:,t] + k3*dt)
            z[:,t+1] = z[:,t] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return {
            'x' : np.array(z[:self.system.dofs,:]),
            'xdot' : np.array(z[self.system.dofs:,:]),
            'z' : np.array(z)
        }
        
class corotational_rk4(rk4):
    """
    Same RK4 integrator but the RHS is rebuilt from K(q), C(q) every step.
    """
    
    def __init__(self, system):
        super().__init__(system)

        # ③ evaluate ω_max from initial tangent and warn if dt too large
        K0, _ = self.system.assemble_KC(np.zeros(self.dofs),
                                        np.zeros(self.dofs))
        # convert sparse to dense once for the eigensolver
        K0 = K0.toarray() if scipy.sparse.issparse(K0) else K0
        eigvals = scipy.linalg.eigvals(K0, self.system.M)
        omega_max = np.sqrt(np.max(np.real(eigvals)))
        dt_user   = self.system.t[1] - self.system.t[0]
        dt_crit   = 0.2 / omega_max        # γ=0.2 safe for RK4

        if dt_user > dt_crit:
            warnings.warn(
                f"Time-step {dt_user:.3e}s exceeds RK4 stability limit "
                f"{dt_crit:.3e}s (ω_max={omega_max:.2f}). "
                f"Expect divergence.", UserWarning)


    def ode_f(self, t, z):
        q = z[:self.dofs]
        v = z[self.dofs:]
        
        f_int = self.system.internal_force(q, v)

        # --- large-angle K,C from the system ------------------------------
        # K, C = self.system.assemble_KC(q, v)

        # --- external force at this time-point ----------------------------
        if self.system.f is None:
            f_ext = np.zeros(self.dofs)
        else:
            t_id  = np.argmin(np.abs(self.t - t))
            f_ext = self.system.f[:, t_id]

        # --- acceleration -------------------------------------------------
        a = np.linalg.solve(self.system.M, 
                            f_ext - f_int)

        return np.concatenate((v, a))

    def sim(self, time, z0):
        ns   = len(time)
        dt = time[1] - time[0]
        dofs = self.system.dofs
        z    = np.zeros((2*dofs, ns))
        acc  = np.zeros((dofs, ns))        # ← store a(t)

        z[:, 0] = z0
        for k in range(ns-1):
            t = time[k]
            k1 = self.ode_f(t,          z[:, k]);       a1 = k1[dofs:]
            k2 = self.ode_f(t+dt/2, z[:, k] + k1*dt/2); a2 = k2[dofs:]
            k3 = self.ode_f(t+dt/2, z[:, k] + k2*dt/2); a3 = k3[dofs:]
            k4 = self.ode_f(t+dt,   z[:, k] + k3*dt);   a4 = k4[dofs:]
            z[:, k+1] = z[:, k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

            # any RK flavour: take the last stage as a(t+dt)
            acc[:, k+1] = a4

        return {'x':     z[:dofs],
                'xdot':  z[dofs:],
                'acc':   acc,
                'z':     z}
    