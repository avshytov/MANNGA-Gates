import numpy as np
import pylab as pl
import sys

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


def view_data(fname):
    d = np.load(fname)
    for k in d.keys():
        print (k)

    I_A = d['I_A']
    I_B = d['I_B']
    O   = d['O']
    V   = d['V']
    D   = d['D']
    I_vals = I_A[:, 0] #d['I_vals']
    logic_low_min = d['Logic_low_min']
    logic_low_max = d['Logic_low_max']
    logic_high_min = d['Logic_high_min']
    logic_high_max = d['Logic_high_max']

    w_A = d['w_A']
    w_B = d['w_B']
    w_P = d['w_P']
    w_L = d['w_L']
    w_O = d['w_O']
    print ("w_A = ", w_A)
    print ("w_B = ", w_B)
    print ("w_P = ", w_P)
    print ("w_L = ", w_L)
    print ("w_O = ", w_O)

    omega = d['omega']
    Delta_R = d['Delta_R']
    Delta_L = d['Delta_L']
    Omega_0 = d['Omega_0']
    Gamma_0 = d['Gamma_0']
    v = d['v']
    lambda_nl = d['lambda_nl']

    Gamma_R = np.abs(Delta_R)**2 / 2.0 / v
    Gamma_L = np.abs(Delta_L)**2 / 2.0 / v
    Gamma_tot = Gamma_0 + Gamma_R + Gamma_L
    print ("Gamma_R = ", Gamma_R, "Gamma_L = ", Gamma_L)
    print ("Gamma_tot = ", Gamma_tot, "Gamma_0 = ", Gamma_0)

    print ("Operating frequency: ", omega)
    

    I_act = d['I_act']
    O_act = d['O_act']
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

    # Logical output
    pl.figure()
    pl.pcolormesh(I_A, I_B, D, cmap=logical_cmap, vmin=-1.0, vmax=1.0)
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    cb.set_ticks([-0.66, 0, 0.66], labels=['INV', 'FALSE', 'TRUE'])
    # Label the range boundaries
    pl.plot([min(I_vals), max(I_vals)],
        [logic_low_min, logic_low_min], 'k--')
    pl.plot([min(I_vals), max(I_vals)],
        [logic_low_max, logic_low_max], 'k--')
    pl.plot([min(I_vals), max(I_vals)],
        [logic_high_min, logic_high_min], 'k--')
    pl.plot([min(I_vals), max(I_vals)],
        [logic_high_max, logic_high_max], 'k--')

    pl.plot([logic_low_min, logic_low_min],
        [min(I_vals), max(I_vals)], 'k--')
    pl.plot([logic_high_min, logic_high_min],
        [min(I_vals), max(I_vals)], 'k--')
    pl.plot([logic_low_max, logic_low_max],
        [min(I_vals), max(I_vals)], 'k--')
    pl.plot([logic_high_max, logic_high_max],
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
        [logic_low_min, logic_low_min], 'k--')
    pl.plot([min(I_vals), max(I_vals)],
        [logic_low_max, logic_low_max], 'k--')
    pl.plot([min(I_vals), max(I_vals)],
        [logic_high_min, logic_high_min], 'k--')
    pl.plot([min(I_vals), max(I_vals)],
        [logic_high_max, logic_high_max], 'k--')

    pl.plot([logic_low_min, logic_low_min],
        [min(I_vals), max(I_vals)], 'k--')
    pl.plot([logic_high_min, logic_high_min],
        [min(I_vals), max(I_vals)], 'k--')
    pl.plot([logic_low_max, logic_low_max],
        [min(I_vals), max(I_vals)], 'k--')
    pl.plot([logic_high_max, logic_high_max],
        [min(I_vals), max(I_vals)], 'k--')
    pl.xlim(min(I_vals), max(I_vals))
    pl.ylim(min(I_vals), max(I_vals))

    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel(r"$I_A$, a.u.")
    pl.ylabel(r"$I_B$, a.u.")
    cb.set_label(r"$V(I_A, I_B)$, a.u.")
    cb.set_ticks([0.25, 0.75], labels=['WRONG', 'RIGHT'])

    pl.show()



for fname in sys.argv[1:]:
    view_data(fname)
