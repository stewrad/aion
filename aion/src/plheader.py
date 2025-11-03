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

# SoF sequence: 127 bits, LFSR of length 7 (primitive poly)
# 7-bit LFSR, taps at positions 7 and 3 (x^7 + x^3 + 1)
sof_seed = [1,1,0,0,1,0,1]
sof_taps = [6,2]            # 0,3,7 prim_poly
sof_sequence = lfsr(sof_seed, sof_taps, 127)
print(f'SoF Sequence: \n{sof_sequence}')

# pilot scrambler sequence 
pls1_seed = [1,1,0,0]
pls1_taps = [3,1]
pilot_pls = lfsr(pls1_seed, pls1_taps, 15)
# pilot sequence: length 16 bits, LFSR of length 4 
pilot_seed = [1,1,0,1]
pilot_taps = [3,0]
pilot_seq = lfsr(pilot_seed, pilot_taps, 15)
pilots_scrambled = pilot_seq * pilot_pls
# print(f'Pilot PLS: {pilot_pls}')
print(f'Pilots: {pilot_seq}')
print(f'Scrambled Pilots: {pilots_scrambled}')

# MCS Indices
# Generate Walsh codes for 8 sequences
N = 16
H = hadamard_matrix(N)
walsh_codes = H  # Each row is a Walsh code of length N
# mcs_scrambled = walsh_codes * mcs_pls
print(f'MCS Indices: \n{walsh_codes}')

# MCS with different LFSR seed or taps 
L = int(np.log2(N*2))
mcs_scrambled = np.zeros_like(walsh_codes)
seed_i = [0,0,0,0,1]
for i in range(N):
    print(f'PLS Seed: {seed_i}')  
    mcs_pls = lfsr(seed_i, taps=[4,2], length=31)
    mcs_scrambled[i,:] = walsh_codes[i,:] * mcs_pls[:N]

    seed_i = seed_increment(seed_i)
print(f'Scrambled MCS Indices: \n{mcs_scrambled}')


