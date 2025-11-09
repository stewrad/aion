'''
Noise Generators for signals created within .sig_gen.{mod_type} generators. 
'''

import numpy as np
from typing import Optional
import warnings

def awgn_gen(
    signal: np.array,
    snr_dB: float=10.0,
    phase_noise_multiplier: float=0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    '''
    Parameters
    ----------
    signal: 
        Input signal to apply noise to. 
    snr_dB: 
        Level of SNR for output signal. 
        Default 10.0
    phase_noise_multiplier:
        Optional - phase noise multiplier to simulate spin from carrier
        Default 0.0
    rng : np.random.Generator, optional
        Source of randomness.  A new default RNG is created if *None*,
        so pass one in for reproducibility.

    Returns 
    -------
    nSig : np.ndarray 
        signal combined with complex AWGN 
    '''

    # Check if rng is None (not provided) or not a Generator
    if rng is None or not isinstance(rng, np.random.Generator):
        # Create a default random number generator
        rng = np.random.default_rng()

    # Calculate signal power (average squared magnitude)
    sigPower = np.mean(np.abs(signal)**2)
    if sigPower == 0:
        warnings.warn("Input signal power is zero. Noise cannot be added based on SNR.")
        noisePower = 0.1 # Ogenerated noise with some default variance?
    else:
        linearSNR = 10 ** (snr_dB / 10)
        noisePower = sigPower / linearSNR

    if np.iscomplexobj(signal):
        # Complex noise: variance split between real and imaginary parts
        # noisePower = E[|N|^2] = E[N_r^2] + E[N_i^2] = 2 * sigma^2
        # So, sigma = sqrt(noisePower / 2)
        noise_std = np.sqrt(noisePower / 2.0)
        noise = noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    else:
        # Real noise: variance = noisePower
        noise_std = np.sqrt(noisePower)
        noise = noise_std * rng.standard_normal(signal.shape)

    nSig = signal + noise

    # Optional: 
    if (phase_noise_multiplier != 0):
        # Apply phase noise
        phase_noise = np.random.randn(len(nSig)) * 0.4     # adjust multiplier for "strength" of phase noise
        nSig = nSig * np.exp(1j*phase_noise)

    return nSig

