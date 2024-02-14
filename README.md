# DynaSim

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
![GitHub repo size](https://img.shields.io/github/repo-size/MarcusHA94/dynasim)
![GitHub contributors](https://img.shields.io/github/contributors/MarcusHA94/dynasim)

<!-- Project name is a `<utility/tool/feature>` that allows `<insert_target_audience>` to do `<action/task_it_does>`. -->

The dynasim package simulates dynamic systems in the form:
$$
\mathbf{M}\ddot{\mathbf{x}} + \mathbf{C}\dot{\mathbf{x}} + \mathbf{K}\mathbf{x} + \mathbf{C}_n g_c(\mathbf{x}, \dot{\mathbf{x}}) + \mathbf{K}_n g_k(\mathbf{x}, \dot{\mathbf{x}}) = \mathbf{f}
$$
where $\mathbf{\Xi}_n g_{\bullet}(\mathbf{x},\dot{\mathbf{x}})$ represents the nonlinear system forces. For example, a 3DOF Duffing oscillator, connected at one end, would have representative nonlinear forces,
$$
\mathbf{K}_n g_n(\mathbf{x}) = \begin{bmatrix}
    k_{n,1} & - k_{n,2} & 0 \\
    0 & k_{n,2} & -k_{n,3} \\
    0 & 0 & k_{n,3}
\end{bmatrix}
\begin{bmatrix}
    x_1^3 \\
    (x_2-x_1)^3 \\
    (x_3 - x_2)^3
\end{bmatrix}
$$

## Installing DynaSim

To install SynaSim, follow these steps:

Linux and macOS:
```
python3 -m pip install dynasim
```

Windows:
```
py -m pip install dynasim
```
## Using DynaSim

### Quickstart Guide

To use DynaSim, here is a quick start guide to generate a 5-DOF oscillating system with some cubic stiffness nonlinearities:

```
import numpy as np
import dynasim

# create required variables
n_dof = 5
nt = 2048
time_span = np.linspace(0, 120, nt)
# time vector of 2048 time points up to 120 seconds

# create vectors of system parameters for sequential MDOF
m_vec = 10.0 * np.ones(n_dof)
c_vec = 1.0 * np.ones(n_dof)
k_vec = 15.0 * np.ones(n_dof)
# imposes every other connection as having an additional cubic stiffness
kn_vec = np.array([25.0 * (i%2) for i in range(n_dof)])

# create nonlinearities
system_nonlin = dynasim.nonlinearities.exponent_stiffness(kn_vec, exponent=3, dofs=n_dof)
# instantiate system and embed nonlinearity
system = dynasim.systems.cantilever(m_vec, c_vec, k_vec, dofs=n_dof, nonlinearity=system_nonlin)

# create excitations and embed to system
system.excitations = [None] * n_dof
system.excitations[-1] = dynasim.actuators.sine_sweep(w_l = 0.5, w_u = 2.0, F0 = 1.0)

# simulate system
data = system.simulate(time_span, z0=None)
```

### Nonlinearities

Three nonlinearities are available, exponent stiffness, exponent damping, and Van der Pol damping

```
dynasim.nonlinearities.exponent_stiffness(kn_vec, exponent=3, dofs=n_dof)
dynasim.nonlinearities.exponent_damping(cn_vec, exponent=0.5, dofs=n_dof)
dynasim.nonlinearities.vanDerPol(cn_vec, dofs=n_dof)
```
These classes contain the $g_k(\mathbf{x}, \dot{\mathbf{x}})$ function.

### Common system classes

There are a currently two system types available for MDOF systems, which are instantiated from vectors of system parameter values:

```
dynasim.systems.mdof_symmetric(m_, c_, k_, dofs, nonlinearity)
dynasim.systems.mdof_cantilever(m_, c_, k_, dofs, nonlinearity)
```

### Actuator classes

The forcing for the system should be a list of actuations, equal in length to the number of DOFs of the system, there many actuation types,
```
dynasim.actuators.sinusoid(...)
dynasim.actuators.white_gaussian(...)
dynasim.actuators.sine_sweep(...)
dynasim.actuators.rand_phase_ms(...)
```

### Totally custom system

One can generate a custom system by instantiating an MDOF system with corresponding modal matrices, but the nonlinearity must also be instantiated and 
```
dynasim.base.mdof_system(M, C, K, Cn, Kn)
```

## Contact

If you want to contact me you can reach me at <mhaywood@ethz.ch>.

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT](<https://opensource.org/license/mit/>).