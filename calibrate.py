import numpy as np

from equations import flops, memory, latency, energy, bytes_moved

FLOP_COEF   = 17_712.0
FLOP_B      = 313_344.0
MEM_COEF    = 56.0
MEM_STATIC  = 1_039_968 * 4
TRAF_COEF   = 188.0
TRAF_STATIC = 1_039_968 * 4

def residuals(theta, S, B, measured):
    F_peak, BW, T_ovh = theta
    pred = np.maximum(
        (FLOP_COEF * B * S**2 + FLOP_B * B) / F_peak,
        (TRAF_COEF * B * S**2 + TRAF_STATIC) / BW
    ) + T_ovh
    return np.log(pred) - np.log(measured)

def residuals_energy(theta, S, B, measured):
    P_idle, e_flop, e_byte = theta
    pred = (P_idle * latency(S, B, theta)
            + flops(S, B) * e_flop
            + bytes_moved(S, B) * e_byte)
    return np.log(pred) - np.log(measured)
