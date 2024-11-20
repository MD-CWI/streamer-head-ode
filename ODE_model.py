#!/usr/bin/env python3

# ODE model for streamer heads. This version is a re-implementation by Jannis
# Teunissen of the code written by Dennis Bouwman during his PhD at CWI,
# Amsterdam.
#
# Repository: https://github.com/MD-CWI/streamer-head-ode
#
# Contributors: Dennis Bouwman, Ute Ebert, Jannis Teunissen

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.optimize import root_scalar, root
import scipy.constants as c
import pickle
import sys
import argparse

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''Solve ODE model for streamer heads.''')
p.add_argument('-Q', type=float, default=0.5,
               help='Charge (C) in units of (4 * pi * eps0) Coulombs')
p.add_argument('-v', type=float, default=7e4,
               help='Velocity (m/s)')
g = p.add_mutually_exclusive_group()
g.add_argument('-R', type=float,
               help='Radius to obtain (m)')
g.add_argument('-E_max', type=float,
               help='Maximal electric field to obtain (V/m)')
p.add_argument('-fixed_I_ph', type=float,
               help='Use fixed value for photoionization source')
p.add_argument('-E_bg', type=float, default=4.5e5,
               help='Background electric field (V/m)')
p.add_argument('-ne_bc', type=float, default=1e10,
               help='Electron density at boundary (1/m3)')
p.add_argument('-T', type=float, default=300,
               help='Gas temperature (K)')
p.add_argument('-p', type=float, default=1.0,
               help='Gas pressure (bar)')
p.add_argument('-frac_O2', type=float, default=0.2,
               help='Fraction of O2')
p.add_argument('-p_q', type=float, default=40e-3,
               help='Photoionization quenching pressure (bar)')
p.add_argument('-photoi_eff', type=float, default=0.075,
               help='Photoionization efficiency')
p.add_argument('-L_factor', type=float, default=1.0,
               help='Scaling factor for domain length')
p.add_argument('-E_threshold', type=float, default=1.0,
               help='Stop when E < E_threshold * E_k')
p.add_argument('-rtol', type=float, default=1e-5,
               help='Relative tolerance for solvers')
p.add_argument('-compact_output', action='store_true',
               help='Print v, R, Q, E_bg, E_max, n_ch, I_ratio, ne_bc')
p.add_argument('-plot', action='store_true',
               help='Plot solution')
p.add_argument('-save_sol', type=str,
               help='Store solution into this pickle file')
p.add_argument('-alpha_table', type=str,
               default="input/reduced_alpha_phelps_jannis.txt",
               help='File with Townsend ionization coefficient alpha')
p.add_argument('-eta_table', type=str,
               default="input/reduced_eta_phelps_jannis.txt",
               help='File with Townsend attachment coefficient alpha')
p.add_argument('-mu_table', type=str,
               default="input/reduced_mu_phelps_jannis.txt",
               help='File with electron mobility coefficient alpha')

args = p.parse_args()

# Gas number density from ideal gas law
N0 = args.p * 1.0e5 / (c.k * args.T)

# Conversion factor for Townsend to SI units
Td_to_SI = 1e-21 * N0

# Minimal z-coordinate to consider
z_min = 1e-5

# Indices of variables
i_ne, i_ni, i_q = 0, 1, 2

# Load transport data files
TD_alpha = np.loadtxt(args.alpha_table, skiprows=2).T
TD_eta = np.loadtxt(args.eta_table, skiprows=2).T
TD_mu = np.loadtxt(args.mu_table, skiprows=2).T

# Construct functions to interpolate transport data
f_alpha = interp1d(Td_to_SI * TD_alpha[0], TD_alpha[1]*N0,
                   fill_value='extrapolate')
f_eta = interp1d(Td_to_SI * TD_eta[0], TD_eta[1]*N0,
                 fill_value='extrapolate')
f_mu = interp1d(Td_to_SI * TD_mu[0], TD_mu[1]/N0,
                fill_value='extrapolate')

# Apply filter when computing dmu/dE
TD_dmu_dE = np.gradient(TD_mu[1]/N0, Td_to_SI * TD_mu[0])
TD_dmu_dE = savgol_filter(TD_dmu_dE, 15, 2)
f_dmu_dE = interp1d(Td_to_SI * TD_mu[0], TD_dmu_dE,
                    bounds_error=False,
                    fill_value=(TD_dmu_dE[0], TD_dmu_dE[-1]))

# Determine critical field
sol = root_scalar(lambda E: f_alpha(E) - f_eta(E), bracket=[0., 1e7])
E_breakdown = sol.root

# For numerical reasons, it is convenient if numbers are around unity.
# Therefore we use this scaling factor. I_ph represents the number of photons
# produced per unit time.
I_ph_fac = 1e15

# Scaling parameters for root-finding algorithm
R0_scale = 1e-3
E_max_scale = 1e7

# Initial condition
y0 = np.zeros(3)
y0[[i_ne, i_ni, i_q]] = [args.ne_bc, args.ne_bc, args.Q]


def get_E(z, q, E_bg):
    """Get electric field"""
    return E_bg + q/z**2


def get_dE_dz(z, q, dq_dz):
    """Spatial derivative of electric field"""
    return dq_dz/z**2 - 2 * q/z**3


def get_z_crit(Q, E_bg):
    """Solve for location of breakdown field, but avoid negative charge"""
    Q_nonneg = np.maximum(Q, 0.)
    return np.sqrt(Q_nonneg / (E_breakdown - args.E_bg))


def f_absorption(z):
    """Absorption function of Zheleznyak model"""
    xmin = 2.6e3 * args.frac_O2 * args.p
    xmax = 150e3 * args.frac_O2 * args.p
    z = np.maximum(z, 1e-12)    # prevent negative values
    return (np.exp(-xmin * z) - np.exp(-xmax * z))/(z * np.log(xmax/xmin))


def get_S_ph(z, I_ph, R):
    """Get photoionization source term, assuming that all photons come from a
    point source located at z=R/2

    I_ph is the number of photons produced per unit time:
    I_ph = p_q/(p + p_q) * photoi_eff * n_ch * pi * R**2 * v,
    where n_ch * pi * R**2 * v is the ionization produced per unit time
    """
    zstar = np.maximum(z - 0.5 * R, z_min)
    S_ph = abs(I_ph) * I_ph_fac * f_absorption(zstar) / (4 * np.pi * zstar**2)
    return S_ph


def ODE_model_rhs(z, y, I_ph, R):
    """Right-hand side of the ODE model"""
    dy = np.zeros_like(y)
    ne, ni, q = y
    E = get_E(z, q, args.E_bg)
    E_abs = np.abs(E)

    alpha, eta = f_alpha(E_abs), f_eta(E_abs)
    mu, dmu_dE = f_mu(E_abs), f_dmu_dE(E_abs)

    S_ph = get_S_ph(z, I_ph, R) if I_ph > 0 else 0.
    src = (alpha - eta) * mu * E_abs * ne + S_ph
    rho_eps = (ni - ne) * c.e / c.epsilon_0
    dq_dz = rho_eps * z**2
    dE_dz = get_dE_dz(z, q, dq_dz)
    dmu_dz = dmu_dE * dE_dz

    # d/dz [ne, ni, q]
    dy[i_ne] = -1 / (args.v + mu * E) * (src + dmu_dz * E * ne +
                                         mu * rho_eps * ne)
    dy[i_ni] = -src / args.v
    dy[i_q] = dq_dz

    return dy


def get_radius(sol):
    """Determine the radius for an ODE solution"""
    field = get_E(sol.t, sol.y[i_q], args.E_bg)
    return sol.t[np.argmax(field)]


def get_E_max(sol):
    """Determine E_max for an ODE solution"""
    field = get_E(sol.t, sol.y[i_q], args.E_bg)
    return field.max()


def stop_condition(z, y, *other_args):
    """Stop integrating (towards z = 0) when this condition is met"""
    return get_E(z, y[i_q], args.E_bg) - args.E_threshold * E_breakdown


stop_condition.terminal = True
stop_condition.direction = -1.


def solve_fixed_I_ph(y0, I_ph, R=args.R):
    """Integrate ODE using I_ph and R for the photoionization source term"""
    y0[i_q] = abs(y0[i_q])
    z_crit = get_z_crit(y0[i_q], args.E_bg)
    z_max = max(args.L_factor * z_crit, 2 * z_min)

    # We need high accuracy here to later do root finding on solutions
    sol = solve_ivp(ODE_model_rhs, [z_max, z_min],
                    y0, dense_output=True,
                    events=stop_condition, rtol=args.rtol,
                    first_step=1e-6,
                    args=(I_ph, R), method='RK45')
    return sol


def residual_photoi(x0, y0):
    """Return difference between guess and updated guess for I_ph and R"""
    I_ph_guess, R_guess = x0
    sol = solve_fixed_I_ph(y0, I_ph_guess, R_guess)
    n_ch = sol.y[i_ni].max()
    R = get_radius(sol)
    I_ph = args.p_q / (1 + args.p_q) * args.photoi_eff * \
        n_ch * np.pi * R**2 * args.v / I_ph_fac
    return (I_ph - I_ph_guess)/max(I_ph, 1.0), (R - R_guess)/R0_scale


def residual_R(Q_guess, y0, I_ph):
    """Return relative difference between obtained radius and goal radius"""
    y0[i_q] = Q_guess
    sol = solve_fixed_I_ph(y0, I_ph)
    R = get_radius(sol)
    return (R - args.R)/R0_scale


def residual_E_max(Q_guess, y0, I_ph, R):
    """Return relative difference between obtained E_max and goal E_max"""
    y0[i_q] = Q_guess
    sol = solve_fixed_I_ph(y0, I_ph, R)
    E_max = get_E_max(sol)
    return (E_max - args.E_max)/E_max_scale


def residual_photoi_R(x0, y0):
    """Return relative differences for I_ph and radius"""
    I_ph_guess = x0[0]
    y0[i_q] = x0[1]
    sol = solve_fixed_I_ph(y0, I_ph_guess)
    n_ch = sol.y[i_ni].max()
    R = get_radius(sol)
    I_ph = args.p_q / (1 + args.p_q) * args.photoi_eff * \
        n_ch * np.pi * R**2 * args.v / I_ph_fac
    return (I_ph - I_ph_guess)/max(I_ph, 1.0), (R - args.R)/R0_scale


def residual_photoi_E_max(x0, y0):
    """Return relative differences for I_ph and E_max"""
    I_ph_guess, R_guess, y0[i_q] = x0
    # print(x0)
    sol = solve_fixed_I_ph(y0, I_ph_guess, R_guess)
    # print("done")
    n_ch = sol.y[i_ni].max()
    E_max = get_E_max(sol)
    R = get_radius(sol)
    I_ph = args.p_q / (1 + args.p_q) * args.photoi_eff * \
        n_ch * np.pi * R**2 * args.v / I_ph_fac
    residual = [(I_ph - I_ph_guess)/max(I_ph, 1.0), (R - R_guess)/R0_scale,
                (E_max - args.E_max)/E_max_scale]
    return residual


def check_convergence(sol):
    if np.abs(sol.fun).max() > 1e-2:
        sys.stderr.write(str(sol) + '\n' + str(args))
        raise ValueError('No convergence, adjust numerical parameters')
    return


if args.R is not None:
    # Solve iterative problem to find solution with given radius
    if args.fixed_I_ph is not None:
        # Fixed photoionization source, determine Q
        tmp = root_scalar(residual_R, args=(y0, args.fixed_I_ph),
                          x0=y0[i_q], bracket=[1e-2*args.Q, 1e4*args.Q],
                          rtol=args.rtol)
        I_ph = args.fixed_I_ph
        y0[i_q] = tmp.root
        sol = solve_fixed_I_ph(y0, I_ph)
    else:
        Q_guesses = np.array([1.0, 0.2, 5, 0.04, 25.]) * args.Q

        for Q in Q_guesses:
            y0[i_q] = Q
            tmp = root(residual_photoi_R, x0=(1., y0[i_q]), args=(y0, ),
                       method='lm', tol=args.rtol, options={'eps': 5e-3})
            if np.abs(tmp.fun).max() < 1e-2:
                break

        check_convergence(tmp)

        I_ph = tmp.x[0]
        y0[i_q] = tmp.x[1]
        sol = solve_fixed_I_ph(y0, I_ph)
elif args.E_max is not None:
    if args.fixed_I_ph is None:
        Q_guesses = np.array([1.0, 0.2, 5, 0.04, 25.]) * args.Q

        for Q in Q_guesses:
            y0[i_q] = Q
            tmp = root(residual_photoi_E_max, x0=(1., 5e-4, y0[i_q]),
                       args=(y0, ), method='lm', tol=args.rtol,
                       options={'eps': 5e-3})
            if np.abs(tmp.fun).max() < 1e-2:
                break

        check_convergence(tmp)

        I_ph, R, y0[i_q] = tmp.x
        sol = solve_fixed_I_ph(y0, I_ph, R)
    elif args.fixed_I_ph == 0.0:
        # Fixed photoionization source, determine Q
        tmp = root_scalar(residual_E_max, args=(y0, args.fixed_I_ph, 0.0),
                          x0=y0[i_q], bracket=[1e-2*args.Q, 1e4*args.Q],
                          rtol=args.rtol)
        I_ph = args.fixed_I_ph
        y0[i_q] = tmp.root
        sol = solve_fixed_I_ph(y0, I_ph)
    else:
        raise ValueError('fixed_I_ph > 0 not supported')
else:
    if args.fixed_I_ph is not None:
        # Solve for fixed Q and I_ph
        I_ph = args.fixed_I_ph
        sol = solve_fixed_I_ph(y0, I_ph)
    else:
        I_ph, R = 1.0, 5e-4
        tmp = root(residual_photoi, x0=(I_ph, R), args=(y0, ),
                   method='lm', tol=args.rtol, options={'eps': 5e-3})
        I_ph, R = tmp.x
        sol = solve_fixed_I_ph(y0, I_ph, R)

n_ch = sol.y[i_ni].max()
field = get_E(sol.t, sol.y[i_q], args.E_bg)
E_max = field.max()
R = get_radius(sol)

current_ratio = 4 * (E_max/args.E_bg - 1) * args.v/R * c.epsilon_0 / \
    (c.e * n_ch * f_mu(args.E_bg))

if args.compact_output:
    # Print v, R, Q, E_bg, E_max, n_ch, I_ratio, ne_bc
    print(f'{args.v:.3e} {R:.3e} {y0[i_q]:.3e} {args.E_bg:.3e} ' +
          f'{E_max:.3e} {n_ch:.3e} {current_ratio:.3e} {args.ne_bc:.3e}')
else:
    print(f'Velocity [m/s]:         {args.v:.3e}')
    print(f'E_bg [V/m]:             {args.E_bg:.3e}')
    print(f'Channel density [m^-3]: {n_ch:.3e}')
    print(f'Radius [m]:             {R:.3e}')
    print(f'E_max [V/m]:            {E_max:.3e}')
    print(f'Q [C/(4 pi eps0)]:      {y0[i_q]:.3e}')
    print(f'Current ratio:          {current_ratio:.3e}')


if args.save_sol:
    with open(args.save_sol, 'wb') as f:
        sol['E'] = get_E(sol.t, sol.y[i_q], args.E_bg)
        sol['q'] = sol.y[i_q] * 4 * np.pi * c.epsilon_0
        sol['z'] = sol.t
        sol['ne'] = sol.y[i_ne]
        sol['ni'] = sol.y[i_ni]
        pickle.dump(sol, f)
    print(f'Solution stored in {args.save_sol} (pickle file)')

if args.plot:
    fig, ax = plt.subplots(4, layout='constrained', sharex=True)
    ax[0].plot(sol.t, sol.y[i_ne], label='n_e')
    ax[0].plot(sol.t, sol.y[i_ni], label='n_i')
    ax[0].plot(sol.t, sol.y[i_ni] - sol.y[i_ne], label='n_i - n_e')
    ax[1].set_ylabel('(m^-3)')
    ax[0].legend()
    ax[1].plot(sol.t, sol.y[i_q] * 4 * np.pi * c.epsilon_0)
    ax[1].set_ylabel('q (C)')
    ax[2].plot(sol.t, get_E(sol.t, sol.y[i_q], args.E_bg))
    ax[2].set_ylabel('E (V/m)')
    ax[3].plot(sol.t, get_S_ph(sol.t, I_ph, R), marker='.')
    ax[3].set_ylabel(r'$S_{ph}$ (m$^{-3}$ s$^{-1}$)')
    ax[3].set_xlabel('z (m)')
    plt.show()
