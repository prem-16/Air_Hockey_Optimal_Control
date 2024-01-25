import numpy as np
from casadi import SX, vertcat

# Start with an empty NLP
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# "Lift" initial conditions
Xk = SX.sym('X0', 3)
w += [Xk]
lbw += 0    # set initial state
ubw += 0    # set initial state
w0 += 0     # set initial state

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = SX.sym('U_' + str(k),14)
    w   += [Uk]
    lbw += [-1]
    ubw += [1]
    w0  += [0]

    # Integrate till the end of the interval
    Xk_end = F_rk4(Xk, Uk)
    J = J + delta_t * 1/2 * (Xk.T @ Q @ Xk + R * Uk**2) # Complete with the stage cost

    # New NLP variable for state at end of interval
    Xk = SX.sym(f'X_{k+1}', 2)
    w += [Xk]
    lbw += [-np.pi/2, -inf]
    ubw += [2*np.pi, inf]
    w0 += [0, 1]

    # Add equality constraint to "close the gap" for multiple shooting
    g   += [Xk_end-Xk]
    lbg += [0, 0]
    ubg += [0, 0]
J = J + 1/2 * (Xk.T @ Q @ Xk) # Complete with the terminal cost (NOTE it should be weighted by delta_t)

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()