# hw1/equations.py
import numpy as np

FLOP_COEF   = 17_712.0
FLOP_B      = 313_344.0
MEM_COEF    = 56.0
MEM_STATIC  = 1_039_968 * 4
TRAF_COEF   = 188.0
TRAF_STATIC = 1_039_968 * 4


def flops(image_size, batch):
    S, B = np.asarray(image_size, float), np.asarray(batch, float)
    return FLOP_COEF * B * S**2 + FLOP_B * B


def memory(image_size, batch):
    S, B = np.asarray(image_size, float), np.asarray(batch, float)
    return MEM_COEF * B * S**2 + 1424 * B + MEM_STATIC


def bytes_moved(image_size, batch):
    S, B = np.asarray(image_size, float), np.asarray(batch, float)
    return TRAF_COEF * B * S**2 + TRAF_STATIC


def latency(image_size, batch, theta):
    # print(theta)
    S, B = np.asarray(image_size, float), np.asarray(batch, float)
    t_c = flops(S, B)       / theta['F_peak']
    t_m = bytes_moved(S, B) / theta['BW']
    return np.maximum(t_c, t_m) + theta['T_overhead']


def energy(image_size, batch, theta_energy):
    S, B = np.asarray(image_size, float), np.asarray(batch, float)
    t = latency(S, B, theta_energy)
    return (theta_energy['P_idle'] * t
            + flops(S, B)       * theta_energy['e_flop']
            + bytes_moved(S, B) * theta_energy['e_byte'])
