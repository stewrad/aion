#!/usr/bin/env python3
"""
acm_sim.py
----------
Adaptive Coding and Modulation (ACM) simulation harness
using:
 - fec_segmentation.py for frame segmentation
 - enc_dec.py for DVB-S2 LDPC encoding
 - MCS selection based on SNR thresholds
"""

import numpy as np
from fec_segmentation import segment_bits
import enc_dec

# ================================
# DVB-S2 MCS Table (exact copy)
# ================================
MCS_TABLE = [
    # QPSK
    {'name':'QPSK-1/4',   'mod':'QPSK',  'code_rate':1/4, 'snr_threshold':0.5},
    {'name':'QPSK-1/3',   'mod':'QPSK',  'code_rate':1/3, 'snr_threshold':0.8},
    {'name':'QPSK-2/5',   'mod':'QPSK',  'code_rate':2/5, 'snr_threshold':1.1},
    {'name':'QPSK-1/2',   'mod':'QPSK',  'code_rate':1/2, 'snr_threshold':1.5},
    {'name':'QPSK-3/5',   'mod':'QPSK',  'code_rate':3/5, 'snr_threshold':1.8},
    {'name':'QPSK-2/3',   'mod':'QPSK',  'code_rate':2/3, 'snr_threshold':2.1},
    {'name':'QPSK-3/4',   'mod':'QPSK',  'code_rate':3/4, 'snr_threshold':2.5},
    {'name':'QPSK-4/5',   'mod':'QPSK',  'code_rate':4/5, 'snr_threshold':2.8},
    {'name':'QPSK-5/6',   'mod':'QPSK',  'code_rate':5/6, 'snr_threshold':3.0},
    {'name':'QPSK-8/9',   'mod':'QPSK',  'code_rate':8/9, 'snr_threshold':3.5},
    # 8APSK
    {'name':'8APSK-3/5',  'mod':'8APSK','code_rate':3/5, 'snr_threshold':4.5},
    {'name':'8APSK-2/3',  'mod':'8APSK','code_rate':2/3, 'snr_threshold':5.0},
    {'name':'8APSK-3/4',  'mod':'8APSK','code_rate':3/4, 'snr_threshold':5.5},
    {'name':'8APSK-5/6',  'mod':'8APSK','code_rate':5/6, 'snr_threshold':6.0},
    {'name':'8APSK-8/9',  'mod':'8APSK','code_rate':8/9, 'snr_threshold':6.5},
    # 16APSK
    {'name':'16APSK-2/3', 'mod':'16APSK','code_rate':2/3,'snr_threshold':7.5},
    {'name':'16APSK-3/4', 'mod':'16APSK','code_rate':3/4,'snr_threshold':8.0},
    {'name':'16APSK-4/5', 'mod':'16APSK','code_rate':4/5,'snr_threshold':8.5},
    {'name':'16APSK-5/6', 'mod':'16APSK','code_rate':5/6,'snr_threshold':9.0},
    {'name':'16APSK-8/9', 'mod':'16APSK','code_rate':8/9,'snr_threshold':9.5},
    # 32APSK
    {'name':'32APSK-3/4','mod':'32APSK','code_rate':3/4,'snr_threshold':11.0},
    {'name':'32APSK-4/5','mod':'32APSK','code_rate':4/5,'snr_threshold':11.5},
    {'name':'32APSK-5/6','mod':'32APSK','code_rate':5/6,'snr_threshold':12.0},
    {'name':'32APSK-8/9','mod':'32APSK','code_rate':8/9,'snr_threshold':12.5},
]

# ================================
# Helper Functions
# ================================
def select_mcs(snr_db: float):
    """Select appropriate MCS entry based on SNR value."""
    valid = [m for m in MCS_TABLE if snr_db >= m['snr_threshold']]
    if not valid:
        return MCS_TABLE[0]
    return valid[-1]

# ================================
# ACM Simulation Routine
# ================================
def run_acm_sim():
    snr_conditions = [0.5, 2.0, 4.0, 6.0, 8.0, 11.5]  # test cases
    packet_bits = np.random.randint(0, 2, 100000, dtype=np.uint8)

    print("\n=== ACM Simulation Start ===")
    print(f"Total input bits: {len(packet_bits)}")

    # Step 1: Segment into LDPC frame-sized blocks
    segments = segment_bits(packet_bits, frame_size=16200)
    print(f"Segmented into {len(segments)} LDPC-ready K-bit frames")

    # Step 2: Loop over SNR conditions
    for snr_db in snr_conditions:
        mcs = select_mcs(snr_db)
        print(f"\n--- SNR = {snr_db:.1f} dB ---")
        print(f"Selected MCS: {mcs['name']}  (Rate={mcs['code_rate']}, Mod={mcs['mod']})")

        encoded_frames = []
        for u_bits in segments:
            # Ensure input matches encoder K
            u_bits = u_bits[:enc_dec.K]
            if len(u_bits) < enc_dec.K:
                u_bits = np.pad(u_bits, (0, enc_dec.K - len(u_bits)), constant_values=0)

            # LDPC encode
            codeword = (u_bits @ enc_dec.G % 2).astype(np.uint8)
            encoded_frames.append(codeword)

        encoded_bits = np.concatenate(encoded_frames)
        print(f"Encoded {len(encoded_frames)} frames → {len(encoded_bits)} bits ready for modulation")

    print("\n=== ACM Simulation Complete ===")


# ================================
if __name__ == "__main__":
    run_acm_sim()
