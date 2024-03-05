import numpy as np
import pylab as pl
import sys

#  Utility function: solves an equation f(x) = 0 by naive bracketing
#
#  Assumes: a < x < b
#
def solve(f, a, b):
    f_a = f(a)    # function is evaluated at the ends of the bracketing
    f_b = f(b)    # interval
    
    tol = 1e-10   # desired accuracy

    #
    # Bracketing loop. Invariant: f(a) * f(b) < 0, root between a and b
    # 
    while abs(b - a) > tol:  # Stop if the accuracy is sufficient
        
        c = 0.5 * (a + b)    # take the midpoint
        f_c = f(c)           # evaluate f(c)
        
        if f_a * f_c <= 0:   # The root is now between a and c
            b = c
            f_b = f_c
        elif f_b * f_c <= 0: # The root is now between c and b
            a = c
            f_a = f_c
        else:                # This could only happen if f(a) * f(b) > 0
            print ("cannot find root")
            #sys.exit(-1)
            raise Exception("cannot find root")

    # Return the result
    return 0.5 * (a + b)

#
# Utility function to make phase continuous, to avoid jumps between
# -pi and pi
#
def make_phase_continuous(phi):
    out = [phi[0]]                # result
    
    for i in range(1, len(phi)):  # scan over the remaining data points
        phi_cur = phi[i]          # current point
        phi_prev = out[-1]        # last point
        while phi_cur - phi_prev > np.pi:  # phases differ too much, remove 2pi
            phi_cur -= 2.0 * np.pi
        while phi_cur - phi_prev < -np.pi: # ditto, add 2pi when needed
            phi_cur += 2.0 * np.pi
        out.append(phi_cur)

    # Overall shift of the curve, similar to the above
    out = np.array(out)
    while np.sum(out) > np.pi * len(out):  # average phase too large
        out -= 2.0 * np.pi
    while np.sum(out) < -np.pi * len(out): # average phase too small
        out += 2.0 * np.pi
    return out

class NLMode:
    #
    # Nonlinear oscillator obeying the equation
    #
    # i dot (phi) = (omega_0 - i Gamma) phi + lambda_nl |phi|^2 phi + I(t)
    #
    # where lambda_nl is the nonlinearity, omega_0 is the resonant frequency,
    # Gamma is the resonant width (total), and I(t) is the driving signal
    #
    #
    def __init__ (self, omega_0, Gamma, lambda_nl):
        self.omega_0  = omega_0
        self.Gamma  = Gamma
        self.lambda_nl = lambda_nl

    def solve(self, I, omega):
        # Solve for the nonlinear response:
        #
        # (omega - omega_0 + i Gamma - lambda |phi|^2) phi = I
        #
        # Let x = |phi|^2
        #
        # It obeys the equation
        #
        # |omega - omega_0 - lambda x|^2 * x + Gamma^2 x = |I|^2
        #
        omega_0 = self.omega_0
        lambda_nl = self.lambda_nl
        Gamma = self.Gamma
        def f(x):
            f_1 = (omega - omega_0 - lambda_nl * x)**2 * x
            return f_1 + Gamma**2 * x - np.abs(I)**2
        # Solve the cubic equation, between the two limits
        x_a = 0.0
        x_b = np.abs(I)**2 / self.Gamma**2
        x = solve(f, x_a, x_b)
        phi = I / (omega - omega_0 - lambda_nl * x + 1j * Gamma)
        #print ("solve: ", I, omega, phi)
        return phi

#
# Simple test routine for debugging
#
def test_nl_osc():
    omega_0 = 1.0   # arbitrarily chosen resonant frequency
    Gamma = 0.1     # decay rate
    lambda_nl = 0.1 # Scale, so that lambda_nl * 1^2 = Gamma
    nl_osc = NLMode(omega_0, Gamma, lambda_nl)
    pl.figure()
    pl.xlabel(r"Frequency $\omega$")
    pl.ylabel(r"Response $|\varphi| / I$")
    for I in [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        omega = np.linspace(-0.5, 1.5, 1001)
        phi = np.vectorize(lambda o: nl_osc.solve(I, o))(omega)
        pl.plot (omega, np.abs(phi) / I, label=r'$I = %g$' % I)
    pl.legend()
    pl.show()

#test_nl_osc()

class NLResonator:
    #
    #  A nonlinear resonator is an oscillator hybridised with two
    #  one-way channels. The waves in the channels propagate with
    #  the velocity v, and hybridisation is given by Delta_L and Delta_R.
    #  Radiative damping broaden the resonance to
    #
    #  Gamma_tot = Gamma_0 + Gamma_R + Gamma_L
    #
    #  where Gamma_R = |Delta_R|^2 / (2v), ditto Gamma_L
    #
    def __init__ (self, v, Delta_R, Delta_L, Omega_0, Gamma_0, lambda_nl):
        self.v = v # speed of the wave in the channel
        self.Delta_R = complex(Delta_R) # hybridisation to the right mover
        self.Delta_L = complex(Delta_L) # ditto right mover
        self.Omega_0 = Omega_0      # resonant frequency
        self.Gamma_0 = Gamma_0      # dissipative linewidth
        self.lambda_nl = lambda_nl  # nonlinearity
        self.Gamma_L = np.abs(Delta_L)**2 / 2.0 / v # linewidth ->L
        self.Gamma_R = np.abs(Delta_R)**2 / 2.0 / v # linewidth ->R
        self.Gamma_tot = Gamma_0 + self.Gamma_L + self.Gamma_R # total
                                                               # linewidth
        print ("Gamma_R = ", self.Gamma_R, "Gamma_L = ", self.Gamma_L)
        print ("Gamma_tot = ", self.Gamma_tot)
        # initialise the nonlinear oscillator with the total linewidth
        self.nl_mode = NLMode(self.Omega_0, self.Gamma_tot, self.lambda_nl)

    def solve(self, I_R, I_L, omega):
        # The response for the wave I_L arriving from the left
        # and the wave I_R arriving from the right:
        #
        # The effective driving of the oscillator is then
        #
        # I = I_R * Delta_R.conj() + I_L * Delta_L.conj()
        #
        # Note that the delayed phases (between the sources and the gate)
        # are assumed to be included into I_R and I_L.
        #

        I_phi = I_R * self.Delta_R.conjugate() + I_L * self.Delta_L.conjugate()
        # The response of the nonlinear mode:
        phi = self.nl_mode.solve(I_phi, omega)
        # wave transmitted to the right:
        O_R = I_R - 1j * self.Delta_R / self.v * phi
        # wave to the left:
        O_L = I_L - 1j * self.Delta_L / self.v * phi
        return O_R, O_L

    #
    # Activation function (transmission) for a given frequency
    #
    def get_activation(self, I_vals, omega):
        OR_vals = np.vectorize(lambda I:
                               self.solve(I, 0.0, omega)[0])(I_vals)
        OL_vals = np.vectorize(lambda I:
                               self.solve(0.0, I, omega)[1])(I_vals)
        return OR_vals, OL_vals

#
# Test routine for debugging and/or inspection
# 
def test_nl_res():
    Omega_0 = 1.0       # arbitrarily chosen resonant frequency
    Gamma_0 = 0.005     # decay rate
    lambda_nl = 0.1     # nonlinearity
    v = 1.0             # speed of the wave
    Delta_R = 0.2
    Delta_L = 0.15
    nl_res = NLResonator(v, Delta_R, Delta_L, Omega_0, Gamma_0, lambda_nl)
    pl.figure()
    pl.xlabel(r"Frequency $\omega$")
    pl.ylabel(r"Transmissivity $T(\omega; I)$")
    ax_T = pl.gca()
    pl.figure()
    pl.xlabel(r"Frequency $\omega$")
    pl.ylabel(r"Transmissivity phase $arg T(\omega; I)$")
    ax_Tphase = pl.gca()
    pl.figure()
    pl.xlabel(r"Frequency $\omega$")
    pl.ylabel(r"Reflectivity $R(\omega; I)$")
    ax_R = pl.gca()
    pl.figure()
    pl.xlabel(r"Re $T_{fw}(\omega)$")
    pl.ylabel(r"Im $T_{fw}(\omega)$")
    pl.gca().set_aspect('equal', 'box')
    ax_Tsmith = pl.gca()
    pl.figure()
    pl.xlabel(r"Re $T_{bk}(\omega)$")
    pl.ylabel(r"Im $T_{bk}(\omega)$")
    ax_Tsmith_rev = pl.gca()
    pl.gca().set_aspect('equal', 'box')

    for I in [0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        omega = np.linspace(0.5, 1.5, 2001)
        O_RtoR = np.vectorize(lambda o: nl_res.solve(I, 0.0, o)[0])(omega)
        O_RtoL = np.vectorize(lambda o: nl_res.solve(I, 0.0, o)[1])(omega)
        O_LtoR = np.vectorize(lambda o: nl_res.solve(0.0, I, o)[0])(omega)
        O_LtoL = np.vectorize(lambda o: nl_res.solve(0.0, I, o)[1])(omega)
        p = ax_T.plot (omega, np.abs(O_RtoR) / I, label=r'$I = %g$' % I)
        #ax_T.plot(omega, np.abs(O_LtoL) / I, '--', color=p[0].get_color())
        p = ax_Tphase.plot (omega, make_phase_continuous(np.angle(O_RtoR)),
                            label=r'$I = %g$' % I)
        #ax_Tphase.plot(omega, make_phase_continuous(np.angle(O_LtoL)),
        #               '--', color=p[0].get_color())
        p = ax_R.plot (omega, np.abs(O_RtoL) / I, label=r'$I = %g$' % I)
        ax_R.plot(omega, np.abs(O_LtoR) / I, '--', color=p[0].get_color())
        p = ax_Tsmith.plot(O_RtoR.real / I, O_RtoR.imag / I,
                       label=r'$I = %g$' % I)
        p = ax_Tsmith.plot(O_RtoR.real / I, O_RtoR.imag / I, '--',
                           color=p[0].get_color(),
                       label=r'$I = %g$' % I)
        ax_Tsmith_rev.plot(O_LtoL.real / I, O_LtoL.imag / I,
                       label=r'$I = %g$' % I)
    ax_T.legend()
    ax_R.legend()
    ax_Tsmith.legend()
    ax_Tsmith_rev.legend()

    pl.figure()
    pl.xlabel(r"Input signal $I$")
    pl.ylabel(r"Transmission $|T|$")
    ax_Tabs = pl.gca()
    pl.figure()
    pl.xlabel(r"Input signal $I$")
    pl.ylabel(r"Transmitted signal $|O|$")
    ax_Oabs = pl.gca()
    pl.figure()
    pl.xlabel(r"Input signal $I$")
    pl.ylabel(r"Phase of transmission $arg (O)$")
    ax_Ophase = pl.gca()
    for omega_in in [1.0, 1.005, 1.01, 1.015, 1.02,
                     0.995, 0.99, 0.985, 0.98]:
        I_vals = np.linspace(0.001, 0.3, 1001)
        o = omega_in
        O_RtoR = np.vectorize(lambda I: nl_res.solve(I, 0.0, o)[0])(I_vals)
        O_RtoL = np.vectorize(lambda I: nl_res.solve(I, 0.0, o)[1])(I_vals)
        O_LtoR = np.vectorize(lambda I: nl_res.solve(0.0, I, o)[0])(I_vals)
        O_LtoL = np.vectorize(lambda I: nl_res.solve(0.0, I, o)[1])(I_vals)
        p = ax_Oabs.plot(I_vals, np.abs(O_RtoR),
                         label=r'$\omega = %g$' % omega_in)
        #ax_Oabs.plot(I_vals, np.abs(O_LtoL), '--', color=p[0].get_color())
        p = ax_Ophase.plot(I_vals, make_phase_continuous(np.angle(O_RtoR)),
                           label=r'$\omega = %g$' % omega_in)
        #ax_Ophase.plot(I_vals, make_phase_continuous(np.angle(O_LtoL)),
        #               '--', color=p[0].get_color())
        p = ax_Tabs.plot(I_vals, np.abs(O_RtoR) / np.abs(I_vals),
                         label=r'$\omega = %g$' % omega_in)
        #ax_Tabs.plot(I_vals, np.abs(O_LtoL) / np.abs(I_vals),
        #             '--', color=p[0].get_color())
    ax_Tabs.legend()
    ax_Oabs.legend()
    ax_Ophase.legend()
    pl.show()

#test_nl_res()

#
# Implementation of the gate, the analog part. The gate is a resonator
# with power/control input attached to the lines (see the slides).
# The weights w_A, w_B, w_P, w_L, w_O define the circuit.
#
class AnalogGate:
    def __init__ (self, nl_res, omega, w_A, w_B, w_P, w_L, w_O):
        self.nl_res = nl_res   # Resonator
        self.omega = omega     # Operating frequency
        self.w_A = w_A         # The weights (see the slides for definition)
        self.w_B = w_B
        self.w_P = w_P
        self.w_L = w_L
        self.w_O = w_O

    #
    # Activation function of the resonator
    #
    def get_activation(self, I_vals):
        OR_vals, OL_vals = self.nl_res.get_activation(I_vals, omega)
        return OR_vals

    #
    # Analog output for given values of inputs A, B, P
    #
    def output(self, I_A, I_B, I_P):
        # The input on the right-moving channel of the resonator
        # This is a linear combination of I_A and I_B with weights
        # w_A and w_B. We also provide the power input in case we need it
        I_R = I_A * self.w_A + I_B * self.w_B + I_P * self.w_P
        # The input in the left-moving channel is entirely from the
        # power port, with weight w_L
        I_L = self.w_L * I_P
        # Outputs calculated from the response
        O_R, O_L = self.nl_res.solve(I_R, I_L, self.omega)
        # Gate output is O_R biased with P, the weight is I_P
        return O_R + self.w_O * I_P

# Debugging the analog gate
def test_analog_gate():
    Omega_0 = 1.0      # arbitrarily chosen resonant frequency
    Gamma_0 = 0.005    # decay rate
    lambda_nl = 0.1    # nonlinearity
    v = 1.0            # speed of the wave
    Delta_R = 0.2
    Delta_L = 0.15
    nl_res = NLResonator(v, Delta_R, Delta_L, Omega_0, Gamma_0, lambda_nl)
    w_A = 0.9
    w_B = 0.9
    w_P = 0.2
    w_L = 0.5j
    w_O =- 0.5j
    omega = 1.005
    gate = AnalogGate(nl_res, omega, w_A, w_B, w_P, w_L, w_O)
    I_vals = np.linspace(0.0, 0.2, 201)
    I_P = 0.08
    I_B, I_A = np.meshgrid(I_vals, I_vals)
    O = np.zeros(np.shape(I_B), dtype=complex)
    for i in range(len(I_vals)):
        for j in range(len(I_vals)):
            O[i, j] = gate.output(I_A[i, j], I_B[i, j], I_P)
    pl.figure()
    pl.pcolormesh(np.abs(I_A), np.abs(I_B), np.abs(O), cmap='jet')
    pl.colorbar()
    pl.gca().set_aspect('equal', 'box')
    pl.show()

test_analog_gate()

#
# Logical ranges are defined to implement logical operations
#
# Analog-to-logic conversion may yield FALSE = 0, TRUE = 1, INVALID = -1
#
# We assume: 
#
#    low_min < FALSE < low_max       high_min < TRUE < high_max
#
#    INVALID < low_min  OR  low_max < INVALID < high_min OR high_max < INVALID
#
class LogicRanges:
    def __init__ (self, low_a, low_b, high_a, high_b):
        self.low_min = min(low_a, low_b)      # initialise the ranges
        self.low_max = max(low_a, low_b)
        self.high_min = min(high_a, high_b)
        self.high_max = max(high_a, high_b)
        if self.high_min < self.low_max:
            raise Exception("min hi < max low: ranges overlap")

        # logical functions we use to check the performance of the gate,
        # defined below
        self.funcs = {
             'AND'   : self.is_AND,
             'OR'    : self.is_OR,
             'XOR'   : self.is_XOR,
             'NAND'  : self.is_NAND,
             'NOR'   : self.is_NOR,  
        }

    #
    # Conversion routine
    #
    def analog_to_logic(self, value):
        if value >= self.high_min and value <= self.high_max: return 1 # TRUE
        if value >= self.low_min  and value <= self.low_max:  return 0 # FALSE
        return -1                                            # INVALID

    #
    # Template for test functions. Compares the analog values
    # val_x, val_y, val_z against logical relation
    #  
    #  z = func (x, y)
    #
    # Returns true if z == func(x, y), otherwise false
    #
    def test_function(self, func, val_x, val_y, val_z):
        x = self.analog_to_logic(val_x)    # Convert to logical values 
        y = self.analog_to_logic(val_y)
        z = self.analog_to_logic(val_z)
        if x == -1 or y == -1: return True # if either input is INVALID,
                                           # the output could be anything
        if (z == -1): return False         # inputs defined, output undefined
                                           #   => does not work
        return (bool(z) == func(bool(x), bool(y)))

    #
    # Check if val_x, val_y, val_z obey z = XOR(x, y)
    #
    def is_XOR(self, val_x, val_y, val_z):
        return self.test_function(lambda x, y: x^y,
                                  val_x,  val_y, val_z)
    
    #
    # Check if val_x, val_y, val_z obey z = AND(x, y)
    #
    def is_AND(self, val_x, val_y, val_z):
        return self.test_function(lambda x, y: x and y,
                                  val_x,  val_y, val_z)
    
    #
    # Check if val_x, val_y, val_z obey z = OR(x, y)
    #
    def is_OR(self, val_x, val_y, val_z):
        return self.test_function(lambda x, y: x or y,
                                  val_x,  val_y, val_z)
    #
    # Check if val_x, val_y, val_z obey z = NOR(x, y)
    #
    def is_NOR(self, val_x, val_y, val_z):
        return self.test_function(lambda x, y: not(x or y),
                                  val_x,  val_y, val_z)
    
    #
    # Check if val_x, val_y, val_z obey z = NAND(x, y)
    #
    def is_NAND(self, val_x, val_y, val_z):
        return self.test_function(lambda x, y: not(x and y),
                             val_x,  val_y, val_z)

    #
    # Selects one of the above functions 
    #
    def is_FUNC(self, func_name):
        return self.funcs[func_name]

#
# Logic gate implementation. A logical gate is an analog gate
# biased with I_P, with logic_ranges applied to inputs and outputs.
# func_name specifies the function that the gate is SUPPOSED to implement.
#
class LogicGate:
    def __init__ (self, analog_gate, logic_ranges, I_P, func_name):
        self.logic_ranges = logic_ranges  # Ranges for TRUE, FALSE
        self.analog_gate = analog_gate    # the gate that does the job
        self.I_P = I_P                    # bias on the power port
        self.func_name = func_name        # the expected function
                                          # e.g. 'NAND'. See LogicRanges
                                          # for available function names.

    #
    # Output of the analog gate for given inputs on A and B
    #
    def analog_output(self, I_A, I_B):
        return self.analog_gate.output(I_A, I_B, self.I_P)

    #
    # Logical output: TRUE, FALSE, INVALID
    #
    def logic_output(self, I_A, I_B):
        O = self.analog_output(I_A, I_B)
        return self.logic_ranges.analog_to_logic(np.abs(O))

    #
    # Check the operation of the gate by scanning over all the values
    # in I_vals for both inputs
    #
    def probe(self, I_vals):
        I_B, I_A = np.meshgrid(I_vals, I_vals)   # Array of I_A and I_B values
        is_func = self.logic_ranges.is_FUNC(self.func_name) # logical function
        O = np.zeros(np.shape(I_A), dtype=complex)  # Analog outputs
        D = np.zeros(np.shape(I_A))                 # Digital outputs
        V = np.zeros(np.shape(I_A))                 # Fidelity
        for i in range(len(I_vals)):
            for j in range(len(I_vals)):
                # Compute the analog output for given inputs
                O[i, j] = self.analog_output(I_A[i, j], I_B[i, j])
                # Compute the logical output
                D[i, j] = self.logic_ranges.analog_to_logic(np.abs(O[i, j]))
                # Verify whether  func(A, B) = output
                V[i, j] = is_func(np.abs(I_A[i, j]),
                                  np.abs(I_B[i, j]),
                                  np.abs(O[i, j]))
        # Return the data array
        return I_A, I_B, O, D, V

    #
    # The routine to visualise operation of the gate, by probing it
    # with I_vals, and then graphing the outputs. 
    #
    def visualize(self, I_vals):
        I_A, I_B, O, D, V = self.probe(I_vals)  # probe the gate outputs
        logic_ranges = self.logic_ranges        # abbreviation

        # Analog output
        pl.figure()
        pl.pcolormesh(I_A, I_B, np.abs(O), cmap='jet',
                      vmin=0.0, vmax=max(I_vals))
        pl.gca().set_aspect('equal', 'box')
        cb = pl.colorbar()
        pl.xlabel(r"$I_A$, a.u.")
        pl.ylabel(r"$I_B$, a.u.")
        cb.set_label(r"$O(I_A, I_B)$, a.u.")
        pl.title("Analogue output")

        logical_cmap_data = (
            (1.0,                 1.0,                 0.6                ),
            (0.2196078431372549,  0.42352941176470588, 0.69019607843137254),
            (0.94117647058823528, 0.00784313725490196, 0.49803921568627452),
        )
        from matplotlib.colors import ListedColormap
        logical_cmap = ListedColormap(logical_cmap_data)
        fidelity_cmap_data = (
            (1.0,                 0.0,                 0.0                ),
            (1.0,                 1.0,                 0.0),
        )
        fidelity_cmap = ListedColormap(fidelity_cmap_data)
        from matplotlib.colors import ListedColormap
        #logical_cmap = ListedColormap(cmap_data)
        # Logical output
        pl.figure()
        pl.pcolormesh(I_A, I_B, D, cmap=logical_cmap, vmin=-1.0, vmax=1.0)
        pl.gca().set_aspect('equal', 'box')
        cb = pl.colorbar()
        cb.set_ticks([-0.66, 0, 0.66], labels=['INV', 'FALSE', 'TRUE'])
        # Label the range boundaries
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.low_min, logic_ranges.low_min], 'k--')
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.low_max, logic_ranges.low_max], 'k--')
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.high_min, logic_ranges.high_min], 'k--')
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.high_max, logic_ranges.high_max], 'k--')

        pl.plot([logic_ranges.low_min, logic_ranges.low_min],
            [min(I_vals), max(I_vals)], 'k--')
        pl.plot([logic_ranges.high_min, logic_ranges.high_min],
            [min(I_vals), max(I_vals)], 'k--')
        pl.plot([logic_ranges.low_max, logic_ranges.low_max],
            [min(I_vals), max(I_vals)], 'k--')
        pl.plot([logic_ranges.high_max, logic_ranges.high_max],
            [min(I_vals), max(I_vals)], 'k--')
        pl.xlim(min(I_vals), max(I_vals))
        pl.ylim(min(I_vals), max(I_vals))
        pl.xlabel(r"$I_A$, a.u.")
        pl.ylabel(r"$I_B$, a.u.")
        cb.set_label(r"$D(I_A, I_B)$, a.u.")
        pl.title("Digital output")

        # Fidelity map
        pl.figure()
        pl.pcolormesh(I_A, I_B, V, cmap=fidelity_cmap, vmin=0, vmax=1.0)
        # Label the range boundaries
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.low_min, logic_ranges.low_min], 'k--')
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.low_max, logic_ranges.low_max], 'k--')
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.high_min, logic_ranges.high_min], 'k--')
        pl.plot([min(I_vals), max(I_vals)],
            [logic_ranges.high_max, logic_ranges.high_max], 'k--')

        pl.plot([logic_ranges.low_min, logic_ranges.low_min],
            [min(I_vals), max(I_vals)], 'k--')
        pl.plot([logic_ranges.high_min, logic_ranges.high_min],
            [min(I_vals), max(I_vals)], 'k--')
        pl.plot([logic_ranges.low_max, logic_ranges.low_max],
            [min(I_vals), max(I_vals)], 'k--')
        pl.plot([logic_ranges.high_max, logic_ranges.high_max],
            [min(I_vals), max(I_vals)], 'k--')
        pl.xlim(min(I_vals), max(I_vals))
        pl.ylim(min(I_vals), max(I_vals))

        pl.gca().set_aspect('equal', 'box')
        cb = pl.colorbar()
        pl.xlabel(r"$I_A$, a.u.")
        pl.ylabel(r"$I_B$, a.u.")
        cb.set_label(r"$V(I_A, I_B)$, a.u.")
        cb.set_ticks([0.25, 0.75], labels=["WRONG", "RIGHT"])

        pl.show()
        

#
# Debugging / visualisation
#
def verify_gate():
    Omega_0 = 1.0      # arbitrarily chosen resonant frequency
    Gamma_0 = 0.005     # decay rate
    lambda_nl = 0.1   # nonlinearity
    v = 1.0            # speed of the wave
    Delta_R = 0.2
    Delta_L = 0.15
    nl_res = NLResonator(v, Delta_R, Delta_L, Omega_0, Gamma_0, lambda_nl)
    #w_A = 0.7
    #w_B = 0.7
    #w_P = 0.1
    #w_L = 0.5
    #w_O = 0.5
    w_A = 0.9
    w_P = 0.2
    w_L = 0.5j
    w_O = -0.5j
    #w_A =  0.640810619928153
    #w_A = 0.5324532908260855
    #w_P = 0.5404092743472492+0.16610178912153356j
    #w_L = -0.3382437474678458-0.1289800391081257j
    #w_O = -0.5277171617305114-0.5982999177996304j
    #w_A = 0.7966929094065044
    #w_P = 0.4241082188561365-0.2600408305898828j
    #w_L = -0.6682358296229151+0.614019687551976j
    #w_O = -0.6225716347968719+0.13579780902993993j
    #w_A = 0.8810016102174743
    #w_P = 0.5023943848310962-0.5005953059810587j
    #w_L = -0.7256009106025378+0.7983597835598264j
    #w_O = -0.663770380115318+0.1730123560282408j
    #w_A = 0.8155436082527137
    #w_P = 0.4309055832954039-0.2989084885135627j
    #w_L = -0.5576790629552625+0.503724803952667j
    #w_O = -0.5766092536835303-0.06018384499222416j
    #w_A = 0.886461671344845
    #w_P = 0.0494642650610782-0.4098956086061584j
    #w_L = -0.04310029475524999+0.5420465897537j
    #w_O = -0.25083205521802804-0.016262966796021062j
    #w_A = 0.9771460341188166
    #w_P = -0.03950446388302105-0.07335402718016973j
    #w_L = -0.04833135648364824+0.10488533331865516j
    #w_O = -0.15331705084469363-0.2716777953531668j
    #w_P *= 0.2/0.08; w_L *=0.2/0.08; w_O *=0.2/0.08
    w_B = w_A
    print ("verify gate: ", w_A, w_B, w_P, w_L, w_O)
    #w_P = 0.702638273364288+0.05545780543499195j
    #w_L = -0.06704440637197209-0.17540235991576678j
    #w_O = -0.7534504687671723-0.2056038696122614j
    #w_A = 0.6303545875769441
    #w_B = 0.6303545875769441
    #w_P = 0.09807976881922155
    #w_L = 0.5582361146218081
    #w_O = 0.6283224952920969
    omega = 1.005
    I_P = 0.08
    analog_gate = AnalogGate(nl_res, omega, w_A, w_B, w_P, w_L, w_O)
    logic_ranges = LogicRanges(0.00, 0.035, 0.045, 0.08)
    logic_gate = LogicGate(analog_gate, logic_ranges, I_P, 'NAND')
    I_vals = np.linspace(0, 0.08, 201)
    logic_gate.visualize(I_vals)
    #I_A, I_B, O, D, V = probe_gate(gate, logic_ranges, I_vals, I_P)
    #visaualize_gate(I_A, I_B, I_vals, O, D, V, logic_ranges)

verify_gate()

# Optimisation of the gate parameters (weights)
#
# GateTuner uses a given non-linear resonator, the operating frequency
# omega, and power input I_P. Its goal is to adjust the weights so that
# a given logical function is performed by the gate.
#
class GateTuner:
    def __init__ (self, nl_res, omega, logic_ranges, I_P, func_name):
        self.nl_res = nl_res                 # Resonator
        self.omega  = omega                  # Signal frequency
        self.logic_ranges = logic_ranges     # Logic
        self.I_P = I_P                       # Power input
        self.func_name = func_name           # function to be implemented,
                                             # e.g. NAND

    #
    # Factory: make a gate with given weights
    #
    def make_gate(self, w_A, w_B, w_P, w_L, w_O):
        analog_gate = AnalogGate(self.nl_res, self.omega,
                                 w_A, w_B, w_P, w_L, w_O)
        logic_gate = LogicGate(analog_gate, self.logic_ranges, self.I_P,
                               self.func_name)
        return logic_gate

    #
    # Resonator activation function, helper function to visualise
    # the operation
    #
    def get_activation(self, I_vals):
        OR_vals, OL_vals = self.nl_res.get_activation(I_vals, self.omega)
        return OR_vals

    #
    # Optimisation. The weights are stored as a vector x.
    # To calibrate performance of the gate, we unpack x,
    # initialise the gate and probe it. The gate is scored
    # based on how many wrong outputs it produced.
    #
    def tune_gate(self, I_vals, w_bounds):
        def unpack_weights(x):    # helper routine, to unpack x -> w
            w_A = x[0]   # unpack the args, assume w_A is real
            w_B = w_A    # assume w_B = w_A 
            w_P = x[1] + 1j * x[2]  # unpack real and imaginary parts
            w_L = x[3] + 1j * x[4]
            w_O = x[5] + 1j * x[6] 
            return w_A, w_B, w_P, w_L, w_O
        def f_score(x):  # function to be minimised
            w_A, w_B, w_P, w_L, w_O = unpack_weights(x)    # unpack weights
            gate = self.make_gate(w_A, w_B, w_P, w_L, w_O) # initialise 
            I_A, I_B, O, D, V = gate.probe(I_vals)         # calibrate
            # count wrong answers:
            # for each pixel, if V = True, we add 0, if V = False, we add 1
            negative_score = np.sum(abs(np.ones(np.shape(V)) - V))
            print ("weights: ", w_A, w_B, w_P, w_L, w_O,
                   "score: ", negative_score)
            return negative_score
        from scipy import optimize
        #
        # Not relevant now
        #
        #w_0 = np.array([ 0.7,  # w_A
        #           # 0.7,  # w_B -- not used now
        #             0.1,  0.0, # w_P
        #             0.5,  0.0, # w_L
        #             0.5,  0.0, # w_O
        #             ])

        #
        # Optimisation method SHOULD NOT rely on gradients: gate score
        # is not a smooth function of its parameters. We use differential
        # evolution here, as it is not gradient-based. 
        #
        # For debugging: add maxiter=3 after w_bounds, to speed up
        # the optimisation (losing the accuracy). 
        #
        #result = optimize.differential_evolution(f_score, w_bounds, maxiter=3)
        result = optimize.differential_evolution(f_score, w_bounds)
        x_opt = result['x']   # optimal weights
        print ("optimization: ", result)
        print ("weights: ", x_opt)
        # Initialise and return a gate with optimal weights. 
        w_A, w_B, w_P, w_L, w_O = unpack_weights(x_opt) 
        gate = self.make_gate(w_A, w_B, w_P, w_L, w_O)
        return gate

# Debug / investigate optimisation
def optimize_gate():
    omega = 1.005  # Operating frequency

    # Parameters of the resonator
    Omega_0 = 1.0       # arbitrarily chosen resonant frequency
    Gamma_0 = 0.005     # decay rate
    lambda_nl = 0.1     # nonlinearity
    v = 1.0             # speed of the wave
    Delta_R = 0.2
    Delta_L = 0.15
    nl_res = NLResonator(v, Delta_R, Delta_L, Omega_0, Gamma_0, lambda_nl)

    # Logical levels, etc
    logic_ranges = LogicRanges(0.0, 0.035, 0.045, 0.08)
    I_P = 0.08
    #I_vals = np.linspace(0.0, 0.08, 21) # for debugging
    I_vals = np.linspace(0.0, 0.08, 101)

    gate_tuner = GateTuner(nl_res, omega, logic_ranges, I_P, 'NAND')

    # Plot the activation function
    I_act = np.linspace(0.002, 0.2, 201)
    O_act = gate_tuner.get_activation(I_act)
    pl.figure()
    pl.plot(I_act, np.abs(O_act))
    pl.xlabel("Input $I$, a.u.")
    pl.ylabel("Transmitted amplitude $|O|$, a.u.")
    pl.figure()
    pl.plot(I_act, make_phase_continuous(np.angle(O_act))/np.pi * 180.0)
    pl.xlabel("Input $I$, a.u.")
    pl.ylabel("Transmitted phase $arg O$, deg")
    pl.figure()
    pl.plot(I_act, np.abs(O_act)/np.abs(I_act))
    pl.xlabel("Input $I$, a.u.")
    pl.ylabel("Transmissivity $|O|/|I|$")
    #pl.show()

    
    # Was used for debugging
    #w_bounds = [(0.976, 0.978), # w_A
    #          #(0.5, 1.0), # w_B -- not used now
    #            (-0.0988, -0.0987), (-0.184, -0.183), # w_P: re, im
    #            (-0.121, -0.12), (0.262, 0.263), # w_L: re, im
    #            (-0.384, -0.383), (-0.680, -0.679)  # w_O: re, im
    #            ]

    #
    # Weight bounds used in optimization.
    # Note that this definition must agree with the unpacking routines
    # defined in GateTuner
    #
    w_bounds = [(0.2, 1.0), # w_A: we assume it is real
                            # overall phase does not matter anyway
               #(0.5, 1.0), # w_B -- not used now
                (-0.8, 0.8), (-0.8, 0.8), # w_P: re, im
                (-0.8, 0.8), (-0.8, 0.8), # w_L: re, im
                (-0.8, 0.8), (-0.8, 0.8)  # w_O: re, im
                ]

    # determine optimal gate parameters
    gate = gate_tuner.tune_gate(I_vals, w_bounds)

    # Save the results and performance data
    I_A, I_B, O, D, V = gate.probe(I_vals)
    np.savez("gate.npz", 
             I_P = I_P, I_vals = I_vals, 
             I_A = I_A, I_B = I_B, O = O, D = D, V = V,
             I_act = I_act, O_act = O_act, 
             Logic_low_min = logic_ranges.low_min,
             Logic_low_max = logic_ranges.low_max,
             Logic_high_min = logic_ranges.high_min,
             Logic_high_max = logic_ranges.high_max,             
             w_A = gate.analog_gate.w_A,
             w_B = gate.analog_gate.w_B,
             w_P = gate.analog_gate.w_P,
             w_L = gate.analog_gate.w_L,
             w_O = gate.analog_gate.w_O,
             omega = gate.analog_gate.omega,
             v     = nl_res.v,
             Delta_L   = nl_res.Delta_L,
             Delta_R   = nl_res.Delta_R,
             Gamma_R   = nl_res.Gamma_R,
             Gamma_L   = nl_res.Gamma_L,
             Gamma_tot = nl_res.Gamma_tot, 
             Omega_0   = nl_res.Omega_0,
             Gamma_0   = nl_res.Gamma_0,
             lambda_nl = nl_res.lambda_nl)
    
    # visualise the performance of the gate
    gate.visualize(I_vals)
    
optimize_gate()
