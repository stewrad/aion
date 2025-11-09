from utils import utils

from typing import Union, List, Optional, Tuple
import warnings
import numpy as np
import dspftw as dsp
import scipy.signal as ss

ArrayLike = Union[np.ndarray, List[Union[int, float]], Tuple[Union[int, float], ...]]

import logging
logger = logging.getLogger(__name__)

# Add this helper function to select optimal radii
def get_dvbs2_apsk_radii(modulation: str, code_rate: float):
    """
    Get DVB-S2 optimal APSK ring radii for given code rate.
    
    Returns (r1, r2) for 16APSK or (r1, r2, r3) for 32APSK
    """
    if modulation == "16APSK":
        # Inner/Outer ring ratio (γ2/γ1) for 16APSK
        if code_rate <= 2/3:
            return (1.0, 2.70)
        elif code_rate <= 3/4:
            return (1.0, 2.85)
        elif code_rate <= 4/5:
            return (1.0, 3.15)
        else:  # 5/6, 8/9
            return (1.0, 3.15)
    
    elif modulation == "32APSK":
        # Inner/Middle/Outer ratios for 32APSK
        if code_rate <= 3/4:
            return (1.0, 2.75, 5.00)
        elif code_rate <= 4/5:
            return (1.0, 2.84, 5.27)
        else:  # 5/6, 8/9
            return (1.0, 2.90, 5.50)
    
    return None

def apsk16_gen(
    bits: int = 1,
    sample_rate: float = 2.0,
    symbol_rate: float = 1.0,
    taps: int = 21,
    rolloff: float = 0.35,
    r1: float = None,  # Inner ring radius
    r2: float = None,  # Outer ring radius
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    '''
    Output 16APSK symbols with RRC filter applied given an input binary array.
    
    16APSK uses two rings: 4 symbols on inner ring, 12 on outer ring.
    Default radii follow DVB-S2 standard for optimal performance.
    
    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate : float
        Number of samples per sec. Must be at least 2.
        Default: 2.0.
    symbol_rate : float
        Number of symbols per sec. Must be at least 1.
        Default: 1.0.
    taps : int
        Number of taps for root-raised cosine filter. Odd is preferred.
        Default: 21
    rolloff : float
        Roll off factor between 0 and 1.
        Default: 0.35
    r1 : float, optional
        Inner ring radius. If None, uses DVB-S2 optimal value.
    r2 : float, optional
        Outer ring radius. If None, uses DVB-S2 optimal value.
    
    Returns
    -------
    t : np.ndarray
        Time stamps in seconds
    signal : np.ndarray
        Complex64 baseband 16APSK samples
    final_phase : float
        Phase (radians) of the last sample
    bw : float
        Approximate bandwidth
    samples_per_symbol : int
        Samples per symbol
    '''
    
    # --------------- Interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)
    
    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    # 16APSK requires 4 bits per symbol
    num_bits = len(bits_arr)
    if num_bits % 4 != 0:
        # Pad with zeros to make multiple of 4
        pad_len = 4 - (num_bits % 4)
        bits_arr = np.concatenate([bits_arr, np.zeros(pad_len, dtype=int)])
        logger.warning(f'Padded {pad_len} zeros to make bits divisible by 4')
    
    # Convert bits to symbols (4 bits = 1 symbol, values 0-15)
    symbols_decimal = utils.bin2decimal(bits_arr.reshape(-1, 4))
    
    # --------------- Parameter validation ---------------
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    if symbol_rate <= 0:
        raise ValueError('symbol_rate must be positive')
    
    samples_per_symbol = int(round(sample_rate / symbol_rate))
    if samples_per_symbol < 2:
        raise ValueError('sample_rate must be at least 2x symbol_rate')
    if abs(samples_per_symbol * symbol_rate - sample_rate) > 1e-6:
        logger.warning(
            f'sample_rate is not an integer multiple of symbol_rate - '
            f'rounded to {samples_per_symbol} samples/symbol.'
        )
    
    if taps <= 0:
        raise ValueError('taps must be positive')
    if rolloff < 0 or rolloff > 1:
        raise ValueError('rolloff must be between 0 and 1')
    
    # --------------- Generate 16APSK constellation ---------------
    # DVB-S2 standard uses 4+12 configuration
    # Optimal radii for different code rates (using typical 3/4 rate values)
    if r1 is None:
        r1 = 1.0  # Inner ring radius (normalized)
    if r2 is None:
        r2 = 2.85  # Outer ring radius (DVB-S2 optimal for 3/4 code rate)
    
    const = np.zeros(16, dtype=complex)
    
    # Inner ring: 4 symbols at pi/4, 3pi/4, 5pi/4, 7pi/4
    for i in range(4):
        angle = np.pi/4 + i * np.pi/2
        const[i] = r1 * np.exp(1j * angle)
    
    # Outer ring: 12 symbols evenly spaced
    for i in range(12):
        angle = np.pi/12 + i * np.pi/6
        const[i + 4] = r2 * np.exp(1j * angle)
    
    # Normalize average power to 1
    avg_power = np.mean(np.abs(const)**2)
    const = const / np.sqrt(avg_power)
    
    # Map symbols to constellation
    constt = const[symbols_decimal]
    
    # --------------- Upsampling ---------------
    # Insert (samples_per_symbol - 1) zeros between each symbol
    symbols_upsampled = np.kron(constt, np.r_[1, np.zeros(samples_per_symbol - 1)])
    
    # --------------- Apply RRC Filter (Pulse Shaping) ---------------
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)
    
    # Generate pulse-shaped 16APSK signal
    signal = ss.convolve(symbols_upsampled, filt, mode='same')
    
    # --------------- Prepare output ---------------
    t = np.arange(len(signal)) / sample_rate
    final_phase = float(np.angle(signal[-1]))
    bw = 0.5 * (1 + rolloff) * symbol_rate
    
    return t, signal, final_phase, bw, samples_per_symbol


def apsk32_gen(
    bits: int = 1,
    sample_rate: float = 2.0,
    symbol_rate: float = 1.0,
    taps: int = 21,
    rolloff: float = 0.35,
    r1: float = None,  # Inner ring radius
    r2: float = None,  # Middle ring radius
    r3: float = None,  # Outer ring radius
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    '''
    Output 32APSK symbols with RRC filter applied given an input binary array.
    
    32APSK uses three rings: 4 symbols on inner, 12 on middle, 16 on outer.
    Default radii follow DVB-S2 standard.
    
    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate : float
        Number of samples per sec. Must be at least 2.
        Default: 2.0.
    symbol_rate : float
        Number of symbols per sec. Must be at least 1.
        Default: 1.0.
    taps : int
        Number of taps for root-raised cosine filter. Odd is preferred.
        Default: 21
    rolloff : float
        Roll off factor between 0 and 1.
        Default: 0.35
    r1, r2, r3 : float, optional
        Ring radii. If None, uses DVB-S2 optimal values.
    
    Returns
    -------
    t : np.ndarray
        Time stamps in seconds
    signal : np.ndarray
        Complex64 baseband 32APSK samples
    final_phase : float
        Phase (radians) of the last sample
    bw : float
        Approximate bandwidth
    samples_per_symbol : int
        Samples per symbol
    '''
    
    # --------------- Interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)
    
    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    # 32APSK requires 5 bits per symbol
    num_bits = len(bits_arr)
    if num_bits % 5 != 0:
        # Pad with zeros to make multiple of 5
        pad_len = 5 - (num_bits % 5)
        bits_arr = np.concatenate([bits_arr, np.zeros(pad_len, dtype=int)])
        logger.warning(f'Padded {pad_len} zeros to make bits divisible by 5')
    
    # Convert bits to symbols (5 bits = 1 symbol, values 0-31)
    symbols_decimal = utils.bin2decimal(bits_arr.reshape(-1, 5))
    
    # --------------- Parameter validation ---------------
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    if symbol_rate <= 0:
        raise ValueError('symbol_rate must be positive')
    
    samples_per_symbol = int(round(sample_rate / symbol_rate))
    if samples_per_symbol < 2:
        raise ValueError('sample_rate must be at least 2x symbol_rate')
    if abs(samples_per_symbol * symbol_rate - sample_rate) > 1e-6:
        logger.warning(
            f'sample_rate is not an integer multiple of symbol_rate - '
            f'rounded to {samples_per_symbol} samples/symbol.'
        )
    
    if taps <= 0:
        raise ValueError('taps must be positive')
    if rolloff < 0 or rolloff > 1:
        raise ValueError('rolloff must be between 0 and 1')
    
    # --------------- Generate 32APSK constellation ---------------
    # DVB-S2 standard uses 4+12+16 configuration
    # Optimal radii for different code rates (using typical 3/4 rate values)
    if r1 is None:
        r1 = 1.0  # Inner ring radius (normalized)
    if r2 is None:
        r2 = 2.75  # Middle ring radius (DVB-S2 optimal for 3/4 code rate)
    if r3 is None:
        r3 = 5.00  # Outer ring radius (DVB-S2 optimal for 3/4 code rate)
    
    const = np.zeros(32, dtype=complex)
    
    # Inner ring: 4 symbols at pi/4, 3pi/4, 5pi/4, 7pi/4
    for i in range(4):
        angle = np.pi/4 + i * np.pi/2
        const[i] = r1 * np.exp(1j * angle)
    
    # Middle ring: 12 symbols evenly spaced
    for i in range(12):
        angle = np.pi/12 + i * np.pi/6
        const[i + 4] = r2 * np.exp(1j * angle)
    
    # Outer ring: 16 symbols evenly spaced
    for i in range(16):
        angle = np.pi/16 + i * np.pi/8
        const[i + 16] = r3 * np.exp(1j * angle)
    
    # Normalize average power to 1
    avg_power = np.mean(np.abs(const)**2)
    const = const / np.sqrt(avg_power)
    
    # Map symbols to constellation
    constt = const[symbols_decimal]
    
    # --------------- Upsampling ---------------
    # Insert (samples_per_symbol - 1) zeros between each symbol
    symbols_upsampled = np.kron(constt, np.r_[1, np.zeros(samples_per_symbol - 1)])
    
    # --------------- Apply RRC Filter (Pulse Shaping) ---------------
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)
    
    # Generate pulse-shaped 32APSK signal
    signal = ss.convolve(symbols_upsampled, filt, mode='same')
    
    # --------------- Prepare output ---------------
    t = np.arange(len(signal)) / sample_rate
    final_phase = float(np.angle(signal[-1]))
    bw = 0.5 * (1 + rolloff) * symbol_rate
    
    return t, signal, final_phase, bw, samples_per_symbol