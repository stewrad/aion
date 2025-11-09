import threading
import time
import zmq
import numpy as np
import scipy.sparse as sp
from contextlib import contextmanager
import os
import sys
from sionna.phy.fec import utils

# Suppress output helper (from your existing TX)
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# Custom modules 
from fec_segmentation import segment_bits, generate_bitstream_from_packets
from enc_dec import compute_generator_matrix
from plheader import sof_gen, mcs_walsh_gen, pilot_gen, plh_mod
from modems import psk
from channel import awgn
from mcs_config import MCS_TABLE, MCS_LOOKUP, ALIST_MAP, MOD_MAP, select_mcs_for_snr, insert_pilots

# ---------------- ZMQ SETUP -----------------
ADDR = "tcp://0.0.0.0:5555"
ctx = zmq.Context()
socket = ctx.socket(zmq.PUB)
socket.bind(ADDR)
print(f"ZMQ PUB bound to {ADDR}")

# ---------------- NOISE THREAD -----------------
def noise_thread(Fs=4000, frame_len=4000, noise_level=0.05):
    """Constantly stream background AWGN noise"""
    while True:
        noise = (np.random.randn(frame_len) + 1j*np.random.randn(frame_len)) * noise_level
        socket.send(noise.astype(np.complex64).tobytes())
        time.sleep(frame_len / Fs)

# ---------------- BURST THREAD -----------------
def burst_thread(meta, sof, mcs_walsh, pilot_sym, burst_interval_s=1.0):
    """
    Generate bursts based on TX_init and stream them at burst_interval_s
    """
    N_LDPC = int(meta['N_LDPC'])
    NUM_PACKETS = meta['NUM_PACKETS']
    Fs = meta['Fs']
    Rs = meta['Rs']

    # Generate initial bitstream
    bitstream = generate_bitstream_from_packets(NUM_PACKETS)

    while True:
        for snr in meta['SNR_VALUES']:
            # ----- MCS selection -----
            mcs_name = select_mcs_for_snr(snr)
            mcs_entry = MCS_LOOKUP[mcs_name]
            code_rate = mcs_entry['code_rate']
            mod_type = mcs_entry['mod']
            mcs_idx = mcs_entry['idx']

            # ----- Segment bits -----
            segments, _ = segment_bits(bitstream, [mcs_name])

            # ----- LDPC encoding -----
            ALIST_FILE = ALIST_MAP[code_rate]
            with suppress_output():
                alist = utils.load_alist(ALIST_FILE)
                H_dense, _, N, _ = utils.alist2mat(alist)
                M = H_dense.shape[0]
                K = N - M
                G = compute_generator_matrix(H_dense)

            encoded_frames = []
            for frame_bits in segments:
                input_bits = frame_bits[:K] if len(frame_bits) >= K else np.concatenate(
                    [frame_bits, np.zeros(K - len(frame_bits), dtype=np.uint8)]
                )
                codeword = (input_bits @ G % 2).astype(np.uint8)
                encoded_frames.append(codeword)

            # ----- Modulate -----
            mod_func = MOD_MAP[mod_type]
            _, bb, _, _, _ = mod_func(np.concatenate(encoded_frames), symbol_rate=Rs, sample_rate=Fs)

            # ----- Insert pilots and headers -----
            bb_w_pilot = insert_pilots(bb, pilot_sym, data_block=1440, pilot_block=36)
            _, mcs_sym, _, _, _ = plh_mod(mcs_walsh[:, mcs_idx], Rs, Fs)
            symbol_frame = np.concatenate([sof, mcs_sym, bb_w_pilot])

            # ----- Apply AWGN per SNR -----
            noisy_bb = awgn.awgn_gen(symbol_frame, snr)

            # ----- Stream -----
            socket.send(noisy_bb.astype(np.complex64).tobytes())
            print(f"Sent burst: {len(noisy_bb)} samples @ SNR={snr:.1f} dB | Modulation: {mcs_name}")

            # Wait until next burst
            time.sleep(burst_interval_s)

            # Test running idle time between bursts
            idle_time = np.random.uniform(0.1, 0.2) # 50-200ms gap
            noise_fill = np.random.normal(0, 0.01, len(symbol_frame)).astype(np.complex64)
            socket.send(noise_fill.tobytes())
            time.sleep(idle_time)

# ---------------- MAIN -----------------
def start_streaming(meta):
    # Generate common headers / pilots
    sof_seq = sof_gen()
    _, sof, _, _, _ = plh_mod(sof_seq, meta['Rs'], meta['Fs'])
    mcs_walsh = mcs_walsh_gen(32)
    pilots = pilot_gen()
    _, pilot_sym, _, _, _ = plh_mod(pilots, meta['Rs'], meta['Fs'])

    # Start noise thread
    t_noise = threading.Thread(target=noise_thread, args=(meta['Fs'], meta['Fs'], 0.05), daemon=True)
    t_noise.start()

    # Start burst thread
    t_burst = threading.Thread(target=burst_thread, args=(meta, sof, mcs_walsh, pilot_sym, 2.0), daemon=True)
    t_burst.start()

    print("Streaming continuous channel with bursts. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

# ---------------- CONFIG -----------------
if __name__ == "__main__":
    meta = {
        'NUM_PACKETS': 100,
        'SNR_VALUES': [0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 12.0, 6.5, 4.5, 2.5, 0.5],
        'N_LDPC': 16200,
        'Rs': 1000,
        'Fs': 4000
    }

    start_streaming(meta)
