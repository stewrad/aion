"""
ACM Transmission Initialization

This script performs:
- packet_generator.py (packet and bitstream generation)
- fec_segmentation.py (MCS-based bit segmentation for FEC)
- enc_dec.py (DVB-S2 standard LDPC encoding on short frame (N=16200))
- modem.py (Modulation)
"""
import logging
import argparse
from gnuradio import gr, blocks, digital, fec
import numpy as np
# Data Handling
from packet_generator import PacketGenerator
from fec_segmentation import segment_bits, generate_bitstream_from_packets #, MCS_TABLE, MCS_LOOKUP
from enc_dec import compute_generator_matrix
from sionna.phy.fec import utils
import scipy.sparse as sp
from modems import psk, apsk
from plheader import sof_gen, mcs_walsh_gen, pilot_gen, plh_mod
# Channel
from channel import awgn

# Benchmark testing
import time
# Plotting
from vis import sig_plot

# =============================================================================================
# Suppressing output from sionna.phy.fec when generating parity check matrices from alist files
import os
import sys
from contextlib import contextmanager
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
# =============================================================================================

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
    "16APSK": psk.qpsk_gen,
    "32APSK": psk.qpsk_gen,
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
    data_frame: np.ndarray,
    pilots: np.ndarray,
    data_block: int=1440,
    pilot_block: int=36
):
    """
    Insert DVB-S2 style pilot blocks into the modulated symbol stream
    Input: 
    data_frame: np.ndarray
        Modulated data symbols (complex)
    pilots: np.ndarray
        Modulated array of pilot symbols (complex) - will repeat or truncate as needed
    data_block: int
        Number of data symbols between pilot blocks (default 1440)
    pilot_block: int
        Number of pilot symbols per pilot block (default 36)
    Returns: 
    np.ndarray : data + pilots interleaved per DVB-S2 pattern 
    """

    num_data = len(data_frame)
    num_blocks = int(np.ceil(num_data / data_block))
    total_pilots = num_blocks * pilot_block

    # Repeat or trim pilots to fit
    pilots = np.resize(pilots, total_pilots)


    out = []
    pilot_idx = 0
    for i in range(num_blocks):
        start = i * data_block
        end = min(start + data_block, num_data)
        out.append(data_frame[start:end])
        out.append(pilots[pilot_idx:pilot_idx+pilot_block])
        pilot_idx += pilot_block

    return np.concatenate(out)

def TX_init(
    meta: dict,
    sof: np.ndarray,
    mcs_walsh: np.ndarray,
    pilot_sym: np.ndarray
):
    # Set short frame N bit value for each codword 
    N_LDPC = int(meta['N_LDPC'])
    # Set symbol rate and sample rate 
    Rs = meta['Rs']
    Fs = meta['Fs']

    # Step 1: Generate bitstream (data traffic)
    NUM_PACKETS = meta['NUM_PACKETS']
    print("=== ACM Simulation Start ===")
    print(f"Generating {NUM_PACKETS} packets...\n")

    bitstream = generate_bitstream_from_packets(NUM_PACKETS)
    print(f"Total bitstream length: {len(bitstream)} bits\n")

    mod_data = []
    symbol_frame = []
    # snr = meta['SNR_VALUES'][0]
    for snr in meta['SNR_VALUES']:
        print(f"\n--- SNR = {snr:.1f} dB ---")

        # Step 3: Select MCS for this SNR
        mcs_name = select_mcs_for_snr(snr)
        mcs_entry = MCS_LOOKUP[mcs_name]
        code_rate = mcs_entry['code_rate']
        mod_type = mcs_entry['mod']
        mcs_idx = mcs_entry['idx']

        print(f"Selected MCS: {mcs_name} ({mod_type}, R={code_rate:.3f})")

        # Step 4: Segment bits
        # N_LDPC = int(meta['N_LDPC'])
        segments, meta = segment_bits(bitstream, [mcs_name])
        print(f"Generated {len(segments)} frames with K≈{int(round(N_LDPC * code_rate))} bits each.")

        # Step 5: Encode each frame using LDPC
        # Load LDPC parity-check matrix 
        ALIST_FILE = ALIST_MAP[code_rate]
        # Load LDPC parity-check matrix
        with suppress_output():
            alist = utils.load_alist(ALIST_FILE)
            H_dense, _, N, _ = utils.alist2mat(alist)
            M = H_dense.shape[0]
            K = N - M
            H_sparse = sp.csr_matrix(H_dense, dtype=np.uint8)
            G = compute_generator_matrix(H_dense)

        encoded_frames = []
        for i, frame_bits in enumerate(segments):
            # Pad/truncate to match generator size K
            input_bits = frame_bits[:K] if len(frame_bits) >= K else np.concatenate(
                [frame_bits, np.zeros(K - len(frame_bits), dtype=np.uint8)]
            )
            codeword = (input_bits @ G % 2).astype(np.uint8)
            encoded_frames.append(codeword)

        print(f"Encoded {len(encoded_frames)} codewords (total {len(encoded_frames) * N} bits).")
        # print(f'Modulation bits: {codeword[:10]}')
        # print(f'Codeword type: {type(codeword)}, Size: {len(codeword)}')

        mod_func = MOD_MAP[mod_type]
        _, bb, _, bw, sps = mod_func(codeword, symbol_rate=Rs, sample_rate=Fs)
        print(f'Modulated {len(bb)} Symbols as {mod_type}')

        bb_w_pilot = insert_pilots(bb, pilot_sym, data_block=1440, pilot_block=36)

        _, mcs_sym, _, _, _ = plh_mod(mcs_walsh[:, mcs_idx], Rs, Fs)

        # mod_data.append(mcs_sym, bb)
        symbol_frame = np.concatenate([sof, mcs_sym, bb_w_pilot])
        mod_data.append(symbol_frame)

        print(f'Combined {len(sof)} SoF symbols + {len(mcs_sym)} MCS symbols + {len(bb_w_pilot)} Data w/ Pilot symbols.')

        # # Plotting tests for noisy cbr signal 
        # noisy_bb = awgn.awgn_gen(bb, snr)
        # sig_plot.plot_all(noisy_bb, Fs, Rs, sps)

    mod_data = np.concatenate(mod_data)
    print(f"\nTotal Modulated Symbols: {mod_data.size}")

    print("\n=== ACM Simulation Complete ===")

    return mod_data

# High-resolution (nanosecond prec) to benchmark TX init process 
def timing(
    meta: dict,
    sof: np.ndarray,
    mcs_walsh: np.ndarray,
    pilot_sym: np.ndarray
):
    start = time.perf_counter()

    # Simulate processing
    TX_init(meta, sof, mcs_walsh, pilot_sym)

    elapsed = time.perf_counter() - start
    print(f"\n\nTotal processing time: {elapsed:.6f} seconds")

# Set up basic configuration (can be made configurable)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for verbose output
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)    # for full debug output
logger.setLevel(logging.WARNING)  # for quieter output

def main():
    logging.info("Starting ACM Transmission Initialization...")
    parser = argparse.ArgumentParser(description="Generating IP/UDP packets and preparing to modulate")

    # parser.add_argument("-c", "--config", type=str, required=True, help="The filepath configuration to generate the rf signals from.")
    # parser.add_argument("-s", "--save", type=str, required=True, help="Where to save the rf generated files and their annotations.")

    # args = parser.parse_args()
    # with open(args.config) as f:
    #     config = yaml.safe_load(f)

    # rf_scene_init(args.save, config)

    # ================================
    # Configuration
    # ================================
    NUM_PACKETS = 100
    SNR_VALUES = [1.0, 3.0, 6.0, 9.0, 12.0]  # simulate changing link conditions
    N_LDPC = 16200

    meta = {
        'NUM_PACKETS': 10, 
        'SNR_VALUES': [1.0, 3.0, 6.0, 9.0, 12.0],  # simulate changing link conditions, 
        'N_LDPC': 16200, 
        'Rs': 1000,
        'Fs': 4000
        }

    sof_seq = sof_gen()
    _, sof, _, _, _ = plh_mod(sof_seq, meta['Rs'], meta['Fs'])
    mcs_walsh = mcs_walsh_gen(32)

    pilots = pilot_gen()
    _, pilot_sym, _, _, _ = plh_mod(pilots, meta['Rs'], meta['Fs'])

    print(f'Shape Pilots: {np.shape(pilot_sym)}')

    timing(meta, sof, mcs_walsh, pilot_sym)

# Generate simulated data RF files and simulated human annotated tags
if __name__ == "__main__":
    main()

