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
from packet_generator import PacketGenerator
from fec_segmentation import segment_bits, generate_bitstream_from_packets #, MCS_TABLE, MCS_LOOKUP
from enc_dec import compute_generator_matrix
from sionna.phy.fec import utils
import scipy.sparse as sp
from modems import psk, apsk

import time

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
    {'name':'QPSK-1/4',   'mod':'QPSK', 'code_rate':1/4, 'snr':0.5},
    {'name':'QPSK-1/3',   'mod':'QPSK', 'code_rate':1/3, 'snr':0.8},
    {'name':'QPSK-2/5',   'mod':'QPSK', 'code_rate':2/5, 'snr':1.1},
    {'name':'QPSK-1/2',   'mod':'QPSK', 'code_rate':1/2, 'snr':1.5},
    {'name':'QPSK-3/5',   'mod':'QPSK', 'code_rate':3/5, 'snr':1.8},
    {'name':'QPSK-2/3',   'mod':'QPSK', 'code_rate':2/3, 'snr':2.1},
    {'name':'QPSK-3/4',   'mod':'QPSK', 'code_rate':3/4, 'snr':2.5},
    {'name':'QPSK-4/5',   'mod':'QPSK', 'code_rate':4/5, 'snr':2.8},
    {'name':'QPSK-5/6',   'mod':'QPSK', 'code_rate':5/6, 'snr':3.0},
    {'name':'QPSK-8/9',   'mod':'QPSK', 'code_rate':8/9, 'snr':3.5},
    # 8APSK
    {'name':'8APSK-3/5',  'mod':'8PSK', 'code_rate':3/5, 'snr':4.5},
    {'name':'8APSK-2/3',  'mod':'8PSK', 'code_rate':2/3, 'snr':5.0},
    {'name':'8APSK-3/4',  'mod':'8PSK', 'code_rate':3/4, 'snr':5.5},
    {'name':'8APSK-5/6',  'mod':'8PSK', 'code_rate':5/6, 'snr':6.0},
    {'name':'8APSK-8/9',  'mod':'8PSK', 'code_rate':8/9, 'snr':6.5},
    # 16APSK
    {'name':'16APSK-2/3', 'mod':'16APSK', 'code_rate':2/3, 'snr':7.5},
    {'name':'16APSK-3/4', 'mod':'16APSK', 'code_rate':3/4, 'snr':8.0},
    {'name':'16APSK-4/5', 'mod':'16APSK', 'code_rate':4/5, 'snr':8.5},
    {'name':'16APSK-5/6', 'mod':'16APSK', 'code_rate':5/6, 'snr':9.0},
    {'name':'16APSK-8/9', 'mod':'16APSK', 'code_rate':8/9, 'snr':9.5},
    # 32APSK
    {'name':'32APSK-3/4', 'mod':'32APSK', 'code_rate':3/4, 'snr':11.0},
    {'name':'32APSK-4/5', 'mod':'32APSK', 'code_rate':4/5, 'snr':11.5},
    {'name':'32APSK-5/6', 'mod':'32APSK', 'code_rate':5/6, 'snr':12.0},
    {'name':'32APSK-8/9', 'mod':'32APSK', 'code_rate':8/9, 'snr':12.5},
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

def TX_init(meta):
    # Set short frame N bit value for each codword 
    N_LDPC = int(meta['N_LDPC'])

    # Step 1: Generate bitstream (data traffic)
    NUM_PACKETS = meta['NUM_PACKETS']
    print("=== ACM Simulation Start ===")
    print(f"Generating {NUM_PACKETS} packets...\n")

    bitstream = generate_bitstream_from_packets(NUM_PACKETS)
    print(f"Total bitstream length: {len(bitstream)} bits\n")

    mod_data = []
    # snr = meta['SNR_VALUES'][0]
    for snr in meta['SNR_VALUES']:
        print(f"\n--- SNR = {snr:.1f} dB ---")

        # Step 3: Select MCS for this SNR
        mcs_name = select_mcs_for_snr(snr)
        mcs_entry = MCS_LOOKUP[mcs_name]
        code_rate = mcs_entry['code_rate']
        mod_type = mcs_entry['mod']

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

        Rs = 1000
        Fs = 4000
        mod_func = MOD_MAP[mod_type]
        _, bb, _, bw, sps = mod_func(codeword, symbol_rate=Rs, sample_rate=Fs)
        
        print(f'Modulated {len(bb)} Symbols as {mod_type}')

        mod_data.append(bb)
        # np.append(mod_data, bb)

    mod_data = np.concatenate(mod_data)
    print(f"\nTotal Modulated Symbols: {mod_data.size}")

    print("\n=== ACM Simulation Complete ===")

    return mod_data 

# High-resolution (nanosecond prec) to benchmark TX init process 
def timing(meta):
    start = time.perf_counter()

    # Simulate processing
    TX_init(meta)

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
    ALIST_FILE = "alist/dvbs2_1_2_N16200.alist"

    meta = {
        'NUM_PACKETS': 10, 
        'SNR_VALUES': [1.0, 3.0, 6.0, 9.0, 12.0],  # simulate changing link conditions, 
        'N_LDPC': 16200, 
        'ALIST_FILE': "alist/dvbs2_1_2_N16200.alist",
        }

    timing(meta)

# Generate simulated data RF files and simulated human annotated tags
if __name__ == "__main__":
    main()

