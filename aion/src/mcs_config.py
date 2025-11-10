import numpy as np
from modems import psk, apsk

# ================================
# Define your MCS table
# ================================
MCS_TABLE = [
    # QPSK
    {'name':'QPSK-1/4',   'mod':'QPSK', 'code_rate':1/4, 'snr':0.5, 'idx': 0},
    {'name':'QPSK-1/3',   'mod':'QPSK', 'code_rate':1/3, 'snr':0.8, 'idx': 1},
    {'name':'QPSK-2/5',   'mod':'QPSK', 'code_rate':2/5, 'snr':1.1, 'idx': 2},
    {'name':'QPSK-1/2',   'mod':'QPSK', 'code_rate':1/2, 'snr':1.5, 'idx': 3},
    {'name':'QPSK-3/5',   'mod':'QPSK', 'code_rate':3/5, 'snr':1.8, 'idx': 4},
    {'name':'QPSK-2/3',   'mod':'QPSK', 'code_rate':2/3, 'snr':2.1, 'idx': 5},
    {'name':'QPSK-3/4',   'mod':'QPSK', 'code_rate':3/4, 'snr':2.5, 'idx': 6},
    {'name':'QPSK-4/5',   'mod':'QPSK', 'code_rate':4/5, 'snr':2.8, 'idx': 7},
    {'name':'QPSK-5/6',   'mod':'QPSK', 'code_rate':5/6, 'snr':3.0, 'idx': 8},
    {'name':'QPSK-8/9',   'mod':'QPSK', 'code_rate':8/9, 'snr':3.5, 'idx': 9},
    # 8APSK, 'idx': 0
    {'name':'8APSK-3/5',  'mod':'8PSK', 'code_rate':3/5, 'snr':4.5, 'idx': 10},
    {'name':'8APSK-2/3',  'mod':'8PSK', 'code_rate':2/3, 'snr':5.0, 'idx': 11},
    {'name':'8APSK-3/4',  'mod':'8PSK', 'code_rate':3/4, 'snr':5.5, 'idx': 12},
    {'name':'8APSK-5/6',  'mod':'8PSK', 'code_rate':5/6, 'snr':6.0, 'idx': 13},
    {'name':'8APSK-8/9',  'mod':'8PSK', 'code_rate':8/9, 'snr':6.5, 'idx': 14},
    # 16APSK
    {'name':'16APSK-2/3', 'mod':'16APSK', 'code_rate':2/3, 'snr':7.5, 'idx': 15},
    {'name':'16APSK-3/4', 'mod':'16APSK', 'code_rate':3/4, 'snr':8.0, 'idx': 16},
    {'name':'16APSK-4/5', 'mod':'16APSK', 'code_rate':4/5, 'snr':8.5, 'idx': 17},
    {'name':'16APSK-5/6', 'mod':'16APSK', 'code_rate':5/6, 'snr':9.0, 'idx': 18},
    {'name':'16APSK-8/9', 'mod':'16APSK', 'code_rate':8/9, 'snr':9.5, 'idx': 19},
    # 32APSK
    {'name':'32APSK-3/4', 'mod':'32APSK', 'code_rate':3/4, 'snr':11.0, 'idx': 20},
    {'name':'32APSK-4/5', 'mod':'32APSK', 'code_rate':4/5, 'snr':11.5, 'idx': 21},
    {'name':'32APSK-5/6', 'mod':'32APSK', 'code_rate':5/6, 'snr':12.0, 'idx': 22},
    {'name':'32APSK-8/9', 'mod':'32APSK', 'code_rate':8/9, 'snr':12.5, 'idx': 23},
]

MCS_LOOKUP = {entry['name']: entry for entry in MCS_TABLE}

ALIST_MAP = {
    1/4: 'alist/dvbs2_1_4_N16200.alist',
    1/3: 'alist/dvbs2_1_3_N16200.alist',
    2/5: 'alist/dvbs2_2_5_N16200.alist',
    1/2: 'alist/dvbs2_1_2_N16200.alist',
    3/5: 'alist/dvbs2_3_5_N16200.alist',
    2/3: 'alist/dvbs2_2_3_N16200.alist',
    3/4: 'alist/dvbs2_3_4_N16200.alist',
    4/5: 'alist/dvbs2_4_5_N16200.alist',
    5/6: 'alist/dvbs2_5_6_N16200.alist',
    8/9: 'alist/dvbs2_8_9_N16200.alist',
}

MOD_MAP = {
    "BPSK": psk.bpsk_gen,
    "QPSK": psk.qpsk_gen,
    "8PSK": psk.psk8_gen,
    "16APSK": apsk.apsk16_gen,
    "32APSK": apsk.apsk32_gen,
    "pi2bpsk": psk.pi2bpsk_gen,
}

def select_mcs_for_snr(snr_db: float):
    """
    Adaptive MCS selection based on SNR thresholds.
    """
    candidates = [entry for entry in MCS_TABLE if snr_db >= entry['snr']]
    if not candidates:
        return MCS_TABLE[0]['name']
    return candidates[-1]['name']

def insert_pilots(
    data_samples: np.ndarray,      # RRC-filtered data samples
    pilot_samples: np.ndarray,     # RRC-filtered pilot samples
    samples_per_symbol: int,       # From RRC (Fs/Rs)
    data_symbols: int = 1440,      # DVB-S2: 1440 data symbols
    pilot_symbols: int = 36        # DVB-S2: 36 pilot symbols
):
    """
    Insert DVB-S2 pilot blocks at symbol-level intervals.
    
    Args:
        data_samples: Modulated data (length = num_symbols × samples_per_symbol)
        pilot_samples: Modulated pilots (length should be pilot_symbols × samples_per_symbol)
        samples_per_symbol: Samples per symbol from RRC filter
        data_symbols: Number of DATA SYMBOLS between pilot blocks (DVB-S2: 1440)
        pilot_symbols: Number of PILOT SYMBOLS per block (DVB-S2: 36)
    
    Returns:
        Combined data + pilots with correct DVB-S2 structure
    """
    # Convert symbol counts to sample counts
    data_block_samples = data_symbols * samples_per_symbol   # 1440 × 4 = 5760
    pilot_block_samples = pilot_symbols * samples_per_symbol # 36 × 4 = 144
    
    # Validate pilot length
    expected_pilot_len = pilot_block_samples
    if len(pilot_samples) < expected_pilot_len:
        # Repeat pilots to fill
        pilot_samples = np.tile(pilot_samples, 
                               int(np.ceil(expected_pilot_len / len(pilot_samples))))
    pilot_samples = pilot_samples[:expected_pilot_len]
    
    # Calculate number of blocks
    num_data_samples = len(data_samples)
    num_blocks = int(np.ceil(num_data_samples / data_block_samples))
    
    # Build output
    out = []
    for i in range(num_blocks):
        # Extract data block
        start = i * data_block_samples
        end = min(start + data_block_samples, num_data_samples)
        out.append(data_samples[start:end])
        
        # Append pilot block
        out.append(pilot_samples)
    
    return np.concatenate(out)