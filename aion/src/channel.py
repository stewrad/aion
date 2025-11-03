import numpy as np

def awgn(signal, snr_db):
    snr = 10**(snr_db/10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise
