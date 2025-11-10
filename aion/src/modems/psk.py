from utils import utils

from typing import Union, List, Optional, Tuple
import warnings
import numpy as np
import dspftw as dsp
import scipy.signal as ss

ArrayLike = Union[np.ndarray, List[Union[int, float]], Tuple[Union[int, float], ...]]

import logging
logger = logging.getLogger(__name__)

def bpsk_gen(
    bits: int=1,
    sample_rate: float=2.0,
    symbol_rate: float=1.0,
    taps: int=21,
    rolloff: float=0.35,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Output BPSK symbols with RRC filter applied given an input binary array 

    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate:
        Number of samples per sec. Must be a float of at least 2.
        Default: 2.0.
    symbol_rate:
        Number of symbols per sec. Must be a float of at least 1.
        Default: 1.0.
    taps:
        Number of taps for root-raised cosine filter generator. Must be an integer. 
        An odd integer ensures initial and final taps are at baud boundaries. 
        Default: 21
    rolloff:
        Roll off factor. Must be a float between 0 and 1.
        Default: 0.25
    
    Returns
    -------
    t : np.ndarray of shape (N,)
        Time stamps in seconds, uniformly spaced 1/sample_rate apart. 
    signal : np.ndarray
        Complex64 baseband BPSK samples
    final_phase : float
        Phase (radians) of the last sample - useful for phase-continuous 
        stitching of multiple bursts. 
    '''

    # Check Parameters
    # --------------- interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)

    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    # Must have a positive sampling rate
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    # Must have a positive symbol rate
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

    # Must have a positive number of taps
    if taps <= 0:
        raise ValueError('taps must be positive')
    # Must have a value between 0 and 1 for beta aka rolloff
    if rolloff < 0 or rolloff > 1:
        raise ValueError('Beta must be between 0 and 1')

    # Symbol Mapping (0 -> 1, 1 -> -1) and Interpolation by upSampF
    symbols = np.kron((-1)**bits_arr, np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to BPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped BPSK signal
    signal = ss.convolve(symbols, filt, mode='same')
    t = np.arange(len(signal)) / sample_rate 
    final_phase = float(np.angle(signal[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, signal, final_phase, bw, samples_per_symbol

# def bpsk_demod(
#     inSig: np.ndarray,
# ):

def qpsk_gen(
    bits: int=1,
    sample_rate: float=2.0,
    symbol_rate: float=1.0,
    taps: int=21,
    rolloff: float=0.35,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Output QPSK symbols with RRC filter applied given an input binary array 

    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate:
        Number of samples per sec. Must be a float of at least 2.
        Default: 2.0.
    symbol_rate:
        Number of symbols per sec. Must be a float of at least 1.
        Default: 1.0.
    taps:
        Number of taps for root-raised cosine filter generator. Must be an integer. 
        An odd integer ensures initial and final taps are at baud boundaries. 
        Default: 21
    rolloff:
        Roll off factor. Must be a float between 0 and 1.
        Default: 0.25
    
    Returns
    -------
    t : np.ndarray of shape (N,)
        Time stamps in seconds, uniformly spaced 1/sample_rate apart. 
    signal : np.ndarray
        Complex64 baseband BPSK samples
    final_phase : float
        Phase (radians) of the last sample - useful for phase-continuous 
        stitching of multiple bursts. 
    '''

    # Check Parameters
    # --------------- interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)

    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    if len(bits_arr) % 2 == 0:
        dibit_arr = utils.bin2dibit(bits_arr)
    else:
        bits_arr = np.append(bits_arr, 0)
        dibit_arr = utils.bin2dibit(bits_arr)

    # Must have a positive sampling rate
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    # Must have a positive symbol rate
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

    # Must have a positive number of taps
    if taps <= 0:
        raise ValueError('taps must be positive')
    # Must have a value between 0 and 1 for beta aka rolloff
    if rolloff < 0 or rolloff > 1:
        raise ValueError('Beta must be between 0 and 1')
    
    # Generate a constellation with 0,1,2,3 symbol-to-bit mapping 
    # const = np.exp(2j*np.pi*np.arange(1, 8, 2) / 8.0)
    const = np.exp(2j*np.pi*np.array([1,3,7,5]) / 8.0)
    constt = const[dibit_arr]

    # Symbol Mapping (00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3) and Interpolation by upSampF
    symbols = np.kron(constt**(-1), np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to QPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped QPSK signal
    signal = ss.convolve(symbols, filt, mode='same')
    t = np.arange(len(signal)) / sample_rate 
    final_phase = float(np.angle(signal[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, signal, final_phase, bw, samples_per_symbol

# def qpsk_demod(
#     inSig: np.ndarray,
# ):
    


def psk8_gen(
    bits: int=1,
    sample_rate: float=2.0,
    symbol_rate: float=1.0,
    taps: int=21,
    rolloff: float=0.35,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Output QPSK symbols with RRC filter applied given an input binary array 

    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate:
        Number of samples per sec. Must be a float of at least 2.
        Default: 2.0.
    symbol_rate:
        Number of symbols per sec. Must be a float of at least 1.
        Default: 1.0.
    taps:
        Number of taps for root-raised cosine filter generator. Must be an integer. 
        An odd integer ensures initial and final taps are at baud boundaries. 
        Default: 21
    rolloff:
        Roll off factor. Must be a float between 0 and 1.
        Default: 0.25
    
    Returns
    -------
    t : np.ndarray of shape (N,)
        Time stamps in seconds, uniformly spaced 1/sample_rate apart. 
    signal : np.ndarray
        Complex64 baseband BPSK samples
    final_phase : float
        Phase (radians) of the last sample - useful for phase-continuous 
        stitching of multiple bursts. 
    '''

    # Check Parameters
    # --------------- interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)

    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    oct_arr = utils.bin2octal(bits_arr)

    # Must have a positive sampling rate
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    # Must have a positive symbol rate
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

    # Must have a positive number of taps
    if taps <= 0:
        raise ValueError('taps must be positive')
    # Must have a value between 0 and 1 for beta aka rolloff
    if rolloff < 0 or rolloff > 1:
        raise ValueError('Beta must be between 0 and 1')
    
    # Generate a constellation with 0,1,2,3 symbol-to-bit mapping 
    const = np.exp(2j * np.pi * np.array([7,6,2,0,4,5,1,3]) / 8.0)
    constt = const[oct_arr]

    # Symbol Mapping (00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3) and Interpolation by upSampF
    symbols = np.kron(constt**(-1), np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to QPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped QPSK signal
    signal = ss.convolve(symbols, filt, mode='same')
    t = np.arange(len(signal)) / sample_rate 
    final_phase = float(np.angle(signal[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, signal, final_phase, bw, samples_per_symbol

def pi2bpsk_gen(
    bits: int=1,
    sample_rate: float=2.0,
    symbol_rate: float=1.0,
    taps: int=21,
    rolloff: float=0.35,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Output PI/2-BPSK symbols with RRC filter applied given an input binary array 

    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate:
        Number of samples per sec. Must be a float of at least 2.
        Default: 2.0.
    symbol_rate:
        Number of symbols per sec. Must be a float of at least 1.
        Default: 1.0.
    taps:
        Number of taps for root-raised cosine filter generator. Must be an integer. 
        An odd integer ensures initial and final taps are at baud boundaries. 
        Default: 21
    rolloff:
        Roll off factor. Must be a float between 0 and 1.
        Default: 0.25
    
    Returns
    -------
    t : np.ndarray of shape (N,)
        Time stamps in seconds, uniformly spaced 1/sample_rate apart. 
    signal : np.ndarray
        Complex64 baseband BPSK samples
    final_phase : float
        Phase (radians) of the last sample - useful for phase-continuous 
        stitching of multiple bursts. 
    '''

    # Check Parameters
    # --------------- interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)

    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    # Must have a positive sampling rate
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    # Must have a positive symbol rate
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

    # Must have a positive number of taps
    if taps <= 0:
        raise ValueError('taps must be positive')
    # Must have a value between 0 and 1 for beta aka rolloff
    if rolloff < 0 or rolloff > 1:
        raise ValueError('Beta must be between 0 and 1')

    # Symbol Mapping (0 -> 1, 1 -> -1) and Interpolation by upSampF
    symbols = np.kron((-1)**bits_arr * np.exp(2j*np.pi*(1/4)*np.arange(0, len((-1)**bits_arr))), np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to BPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped BPSK signal
    signal = ss.convolve(symbols, filt, mode='same')
    t = np.arange(len(signal)) / sample_rate 
    final_phase = float(np.angle(signal[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, signal, final_phase, bw, samples_per_symbol

def pi4qpsk_gen(
    bits: int=1,
    sample_rate: float=2.0,
    symbol_rate: float=1.0,
    taps: int=21,
    rolloff: float=0.35,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Output PI/4-QPSK symbols with RRC filter applied given an input binary array 

    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate:
        Number of samples per sec. Must be a float of at least 2.
        Default: 2.0.
    symbol_rate:
        Number of symbols per sec. Must be a float of at least 1.
        Default: 1.0.
    taps:
        Number of taps for root-raised cosine filter generator. Must be an integer. 
        An odd integer ensures initial and final taps are at baud boundaries. 
        Default: 21
    rolloff:
        Roll off factor. Must be a float between 0 and 1.
        Default: 0.25
    
    Returns
    -------
    t : np.ndarray of shape (N,)
        Time stamps in seconds, uniformly spaced 1/sample_rate apart. 
    signal : np.ndarray
        Complex64 baseband BPSK samples
    final_phase : float
        Phase (radians) of the last sample - useful for phase-continuous 
        stitching of multiple bursts. 
    '''

    # Check Parameters
    # --------------- interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)

    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    if len(bits_arr) % 2 == 0:
        dibit_arr = utils.bin2dibit(bits_arr)
    else:
        bits_arr = np.append(bits_arr, 0)
        dibit_arr = utils.bin2dibit(bits_arr)

    # Must have a positive sampling rate
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    # Must have a positive symbol rate
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

    # Must have a positive number of taps
    if taps <= 0:
        raise ValueError('taps must be positive')
    # Must have a value between 0 and 1 for beta aka rolloff
    if rolloff < 0 or rolloff > 1:
        raise ValueError('Beta must be between 0 and 1')
    
    # Generate a constellation with 0,1,2,3 symbol-to-bit mapping 
    const = np.exp(2j*np.pi*np.arange(1, 8, 2) / 8.0)
    constt = const[dibit_arr]

    # Symbol Mapping (00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3) and Interpolation by upSampF
    # symbols = np.kron(constt**(-1), np.r_[1, np.zeros(samples_per_symbol-1)])
    symbols = np.kron(constt*np.exp(2j*np.pi*(1/8)*np.arange(0, len(constt))), np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to QPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped QPSK signal
    signal = ss.convolve(symbols, filt, mode='same')
    t = np.arange(len(signal)) / sample_rate 
    final_phase = float(np.angle(signal[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, signal, final_phase, bw, samples_per_symbol

def sqpsk_gen(
    bits: int=1,
    sample_rate: float=2.0,
    symbol_rate: float=1.0,
    taps: int=21,
    rolloff: float=0.35,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Output Staggered QPSK symbols with RRC filter applied given an input binary array 

    Parameters
    ----------
    bits : int | array-like of {0,1}
        * If an ``int`` → length of the random bit sequence to draw.  
        * If array-like → exact bit sequence (must contain only 0 and 1).
    sample_rate:
        Number of samples per sec. Must be a float of at least 2.
        Default: 2.0.
    symbol_rate:
        Number of symbols per sec. Must be a float of at least 1.
        Default: 1.0.
    taps:
        Number of taps for root-raised cosine filter generator. Must be an integer. 
        An odd integer ensures initial and final taps are at baud boundaries. 
        Default: 21
    rolloff:
        Roll off factor. Must be a float between 0 and 1.
        Default: 0.25
    
    Returns
    -------
    t : np.ndarray of shape (N,)
        Time stamps in seconds, uniformly spaced 1/sample_rate apart. 
    signal : np.ndarray
        Complex64 baseband BPSK samples
    final_phase : float
        Phase (radians) of the last sample - useful for phase-continuous 
        stitching of multiple bursts. 
    '''

    # Check Parameters
    # --------------- interpret / generate bit sequence ---------------
    if isinstance(bits, (int, np.integer)):
        rng = rng or np.random.default_rng()
        bits_arr = rng.integers(0, 2, int(bits), dtype=int)
    else:
        bits_arr = np.asarray(bits, dtype=int)

    if not np.all(np.isin(bits_arr, [0, 1])):
        raise ValueError("bits must be 0/1 values.")
    
    if len(bits_arr) % 2 == 0:
        dibit_arr = utils.bin2dibit(bits_arr)
    else:
        bits_arr = np.append(bits_arr, 0)
        dibit_arr = utils.bin2dibit(bits_arr)

    # Must have a positive sampling rate
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    # Must have a positive symbol rate
    if symbol_rate <= 0:
        raise ValueError('symbol_rate must be positive')
    samples_per_symbol = int(round(sample_rate / symbol_rate))
    if samples_per_symbol < 4:
        warnings.warn(
            f"sample rate ({sample_rate} Hz) < 4xsymbol_rate "
            f"({symbol_rate} Hz).  Signal fidelity may be compromised.",
            UserWarning,
        )
    if abs(samples_per_symbol * symbol_rate - sample_rate) > 1e-6:
        logger.warning(
            f'sample_rate is not an integer multiple of symbol_rate - '
            f'rounded to {samples_per_symbol} samples/symbol.'
        )

    # Must have a positive number of taps
    if taps <= 0:
        raise ValueError('taps must be positive')
    # Must have a value between 0 and 1 for beta aka rolloff
    if rolloff < 0 or rolloff > 1:
        raise ValueError('Beta must be between 0 and 1')
    
    # Generate a constellation with 0,1,2,3 symbol-to-bit mapping 
    const = np.exp(2j*np.pi*np.arange(1, 8, 2) / 8.0)
    constt = const[dibit_arr]

    # Symbol Mapping (00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3) and Interpolation by upSampF
    symbols = np.kron(constt, np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to QPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped QPSK signal
    s = ss.convolve(symbols, filt, mode='same')
    sr = np.real(s)
    si = np.imag(s)
    delayIdx = int(samples_per_symbol/2)
    signal = sr[delayIdx:] + 1j*si[0:len(si)-delayIdx]

    t = np.arange(len(signal)) / sample_rate 
    final_phase = float(np.angle(signal[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, signal, final_phase, bw, samples_per_symbol


