#!/usr/bin/env python3
"""
ACM Simulation Driver

This script ties together:
- packet_generator.py (packet and bitstream generation)
- fec_segmentation.py (MCS-based bit segmentation)
- enc_dec.py (LDPC encoding)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow INFO and WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: disable verbose oneDNN messages

import tensorflow as tf
tf.get_logger().setLevel("ERROR")           # suppress TensorFlow runtime logs

import numpy as np
from packet_generator import PacketGenerator
from fec_segmentation import segment_bits, MCS_TABLE, MCS_LOOKUP, generate_bitstream_from_packets
from enc_dec import compute_generator_matrix
from sionna.phy.fec import utils
import scipy.sparse as sp


# ================================
# Configuration
# ================================
NUM_PACKETS = 100
SNR_VALUES = [1.0, 3.0, 6.0, 9.0, 12.0]  # simulate changing link conditions
N_LDPC = 16200
ALIST_FILE = "alist/dvbs2_1_2_N16200.alist"

# Load LDPC parity-check matrix
alist = utils.load_alist(ALIST_FILE)
H_dense, _, N, _ = utils.alist2mat(alist)
M = H_dense.shape[0]
K = N - M
H_sparse = sp.csr_matrix(H_dense, dtype=np.uint8)
G = compute_generator_matrix(H_dense)

print(f"[LDPC] Loaded parity-check matrix: K={K}, N={N}\n")


# ================================
# Define ACM MCS Selection Function
# ================================
def select_mcs_for_snr(snr_db: float):
    """
    Adaptive MCS selection based on SNR thresholds.
    """
    candidates = [entry for entry in MCS_TABLE if snr_db >= entry['snr_threshold']]
    if not candidates:
        return MCS_TABLE[0]['name']
    return candidates[-1]['name']


# ================================
# Run Simulation
# ================================
print("=== ACM Simulation Start ===")
print(f"Generating {NUM_PACKETS} packets...\n")

# Step 1: Generate bitstream
bitstream = generate_bitstream_from_packets(NUM_PACKETS)
print(f"Total bitstream length: {len(bitstream)} bits\n")

# Step 2: Loop over SNRs
for snr in SNR_VALUES:
    print(f"\n--- SNR = {snr:.1f} dB ---")

    # Select MCS for this SNR
    mcs_name = select_mcs_for_snr(snr)
    mcs_entry = MCS_LOOKUP[mcs_name]
    code_rate = mcs_entry['code_rate']
    mod_type = mcs_entry['mod']

    print(f"Selected MCS: {mcs_name} ({mod_type}, R={code_rate:.3f})")

    # Segment bits
    segments, meta = segment_bits(bitstream, [mcs_name])
    print(f"Generated {len(segments)} frames with K≈{int(round(N_LDPC * code_rate))} bits each.")

    # Step 3: Encode each frame using LDPC
    encoded_frames = []
    for i, frame_bits in enumerate(segments):
        # Pad/truncate to match generator size K
        input_bits = frame_bits[:K] if len(frame_bits) >= K else np.concatenate(
            [frame_bits, np.zeros(K - len(frame_bits), dtype=np.uint8)]
        )
        codeword = (input_bits @ G % 2).astype(np.uint8)
        encoded_frames.append(codeword)

    print(f"Encoded {len(encoded_frames)} codewords (total {len(encoded_frames) * N} bits).")
    print(f'Modulation bits: {codeword[:10]}')

    # Step 4: Optional BER simulation
    noise = np.random.randn(*encoded_frames[0].shape)
    snr_lin = 10 ** (snr / 10)
    received = np.array(encoded_frames[0]) + noise / np.sqrt(snr_lin)
    ber_est = np.mean((received > 0.5) != encoded_frames[0])
    print(f"Estimated BER at {snr:.1f} dB ≈ {ber_est:.3e}")

print("\n=== ACM Simulation Complete ===")
print("Encoded bits ready for modulation stage.")
