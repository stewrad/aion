'''
PHY Header Generation 
'''
import numpy as np

def hadamard_matrix(n):
    '''Recursive Hadamard matrix of size n (n must be power of 2).'''
    if n == 1:
        return np.array([[1]])
    else:
        H_n = hadamard_matrix(n//2)
        top = np.hstack([H_n, H_n])
        bottom = np.hstack([H_n, -H_n])
        return np.vstack([top, bottom])

def lfsr(seed, taps, length):
    '''
    length: number of output bits to generate
    seed: initial state as list of 0/1, length = LFSR length
    taps: feedback tap positions (0-based)
    '''
    sr = seed.copy()
    out = []
    for _ in range(length):
        out.append(sr[-1])
        feedback = 0
        for t in taps:
            feedback ^= sr[t]
        sr = [feedback] + sr[:-1]
    return (-1)**np.array(out)  # +/- 1 symbol mapping

def seed_increment(seed_arr):
    '''
    Increment a binary array representing an integer by 1.
    seed_arr: array-like of 0/1, length L
    Returns new array of same length
    '''
    # Convert to integer
    seed_int = int("".join(str(b) for b in seed_arr), 2)
    # Increment and wrap around
    max_val = 2**len(seed_arr) - 1
    seed_int = (seed_int + 1) % max_val
    # Convert back to binary array
    seed_bin = [int(b) for b in format(seed_int, f"0{len(seed_arr)}b")]
    return seed_bin

import dspftw as dsp
import scipy.signal as ss

def sof_gen():
    # Generate SoF Sequence -----------------------------------
    # SoF sequence: 127 bits, LFSR of length 7 (primitive poly)
    # 7-bit LFSR, taps at positions 7 and 3 (x^7 + x^3 + 1)
    sof_seed = [1,1,0,0,1,0,1]
    sof_taps = [6,2]            # 0,3,7 prim_poly
    sof_sequence = lfsr(sof_seed, sof_taps, 127)

    return sof_sequence

def mcs_walsh_gen(
    N: int=32,
):
    # MCS Indices
    # Generate Walsh codes for N sequences
    N = 32
    H = hadamard_matrix(N)
    walsh_codes = H  # Each row is a Walsh code of length N

    # MCS with different LFSR seed or taps 
    L = int(np.log2(N*2))
    mcs_scrambled = np.zeros_like(walsh_codes)
    seed_i = [0,0,0,0,1]
    for i in range(N):
        # print(f'PLS Seed: {seed_i}')  
        mcs_pls = lfsr(seed_i, taps=[4,2], length=63)
        mcs_scrambled[i,:] = walsh_codes[i,:] * mcs_pls[:N]

        seed_i = seed_increment(seed_i)

    return mcs_scrambled[:24]

def pilot_gen():
    # # pilot scrambler sequence 
    # pls1_seed = [1,1,0,0,0,1]
    # pls1_taps = [5,1]
    # pilot_pls = lfsr(pls1_seed, pls1_taps, 63)
    # pilot sequence: length 64 bits, LFSR of length 6 
    pilot_seed = [1,1,0,1,0,1]
    pilot_taps = [5,0]
    pilot_seq = lfsr(pilot_seed, pilot_taps, 63)
    # pilots_scrambled = pilot_seq * pilot_pls
    # # print(f'Pilot PLS: {pilot_pls}')
    # print(f'Pilots: {pilot_seq}')
    # # print(f'Scrambled Pilots: {pilots_scrambled}')
    # print(f'Scrambled Pilots (to keep): {pilots_scrambled[:36]}')

    return pilot_seq[:36]

def plh_mod(
    symbols: np.ndarray,
    symbol_rate: float,
    sample_rate: float,
    taps: int=21,
    rolloff: float=0.35
):
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

    # Modulate symbols to PI/2-BPSK
    uSym = np.kron(symbols * np.exp(2j*np.pi*(1/4)*np.arange(0, len(symbols))), np.r_[1, np.zeros(samples_per_symbol-1)])

    # Apply RRC Filter to PI/2-BPSK (Pulse Shaping)
    filt = dsp.rrcfiltgen(taps, symbol_rate, sample_rate, rolloff)

    # Generate pulse-shaped PI/2-BPSK signal
    pi2sym = ss.convolve(uSym, filt, mode='same')
    t = np.arange(len(pi2sym)) / sample_rate 
    final_phase = float(np.angle(pi2sym[-1]))

    bw = 0.5 * (1 + rolloff) * symbol_rate

    return t, pi2sym, final_phase, bw, samples_per_symbol
    # return pi2sym

if __name__ == "__main__":
    # SoF sequence: 127 bits, LFSR of length 7 (primitive poly)
    # 7-bit LFSR, taps at positions 7 and 3 (x^7 + x^3 + 1)
    sof_seed = [1,1,0,0,1,0,1]
    sof_taps = [6,2]            # 0,3,7 prim_poly
    sof_sequence = lfsr(sof_seed, sof_taps, 127)
    print(f'SoF Sequence: \n{sof_sequence}')
    print(f'SoF Sequence Type: {type(sof_sequence)}')

    # # pilot scrambler sequence 
    # pls1_seed = [1,1,0,0]
    # pls1_taps = [3,1]
    # pilot_pls = lfsr(pls1_seed, pls1_taps, 15)
    # # pilot sequence: length 16 bits, LFSR of length 4 
    # pilot_seed = [1,1,0,1]
    # pilot_taps = [3,0]
    # pilot_seq = lfsr(pilot_seed, pilot_taps, 15)
    # pilots_scrambled = pilot_seq * pilot_pls
    # # print(f'Pilot PLS: {pilot_pls}')
    # print(f'Pilots: {pilot_seq}')
    # print(f'Scrambled Pilots: {pilots_scrambled}')

    # pilot_gen()

    # # MCS Indices
    # # Generate Walsh codes for 16 sequences
    # N = 32
    # H = hadamard_matrix(N)
    # walsh_codes = H  # Each row is a Walsh code of length N
    # # mcs_scrambled = walsh_codes * mcs_pls
    # # print(f'MCS Indices: \n{walsh_codes}')

    # # MCS with different LFSR seed or taps 
    # L = int(np.log2(N*2))
    # mcs_scrambled = np.zeros_like(walsh_codes)
    # seed_i = [0,0,0,0,1]
    # for i in range(N):
    #     # print(f'PLS Seed: {seed_i}')  
    #     mcs_pls = lfsr(seed_i, taps=[4,2], length=63)
    #     mcs_scrambled[i,:] = walsh_codes[i,:] * mcs_pls[:N]

    #     seed_i = seed_increment(seed_i)
    # # print(f'Scrambled MCS Indices: \n{mcs_scrambled}')
    # # print(f'First 24 Walsh Codes: {mcs_scrambled[:24]}')

    # # for i in range(24):
    # #     n = mcs_scrambled[i]
    # #     print(f'Code {i:00d}: MCS={n}')

    # idx = 0
    # print(f'MCS Index {idx}: {mcs_scrambled[:, idx]}')

