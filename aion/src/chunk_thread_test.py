import threading
import time
import zmq
import numpy as np
import scipy.sparse as sp
from contextlib import contextmanager
import os
import sys
from sionna.phy.fec import utils

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

# ---------------- REAL-TIME STREAMING HELPERS -----------------
def stream_samples(samples, Fs):
    """
    Stream samples at real-time pace according to Fs
    """
    samples = samples.astype(np.complex64)
    chunk_size = 1024  # smaller chunks to approximate real-time
    total_samples = len(samples)
    sent = 0
    t0 = time.perf_counter()
    while sent < total_samples:
        chunk = samples[sent:sent+chunk_size]
        socket.send(chunk.tobytes())
        sent += len(chunk)
        # compute how long we should have taken to send these samples
        expected_time = sent / Fs
        elapsed = time.perf_counter() - t0
        sleep_time = expected_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# ---------------- NOISE THREAD -----------------
def noise_thread(Fs=4000, noise_level=0.5):
    """
    Continuously stream background AWGN noise at real-time
    """
    chunk_len = Fs  # 1 second of noise
    while True:
        noise = (np.random.randn(chunk_len) + 1j*np.random.randn(chunk_len)) * noise_level
        stream_samples(noise, Fs)

# ---------------- BURST THREAD -----------------
def burst_thread(meta, sof, mcs_walsh, pilot_sym, burst_interval_s=1.0):
    """
    Generate and stream DVB-S2 bursts in real-time
    """
    N_LDPC = int(meta['N_LDPC'])
    NUM_PACKETS = meta['NUM_PACKETS']
    Fs = meta['Fs']
    Rs = meta['Rs']

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

            # ----- Stream in real-time -----
            print(f"Starting transmission: {len(noisy_bb)} samples @ SNR={snr:.1f} dB")
            stream_samples(noisy_bb, Fs)
            print(f"End of transmission: {len(noisy_bb)} samples @ SNR={snr:.1f} dB")

            # Wait until next burst (maintain spacing)
            time.sleep(burst_interval_s)

# ---------------- MAIN -----------------
def start_streaming(meta):
    sof_seq = sof_gen()
    _, sof, _, _, _ = plh_mod(sof_seq, meta['Rs'], meta['Fs'])
    mcs_walsh = mcs_walsh_gen(32)
    pilots = pilot_gen()
    _, pilot_sym, _, _, _ = plh_mod(pilots, meta['Rs'], meta['Fs'])

    # Start noise thread
    t_noise = threading.Thread(target=noise_thread, args=(meta['Fs'], 0.05), daemon=True)
    t_noise.start()

    # Start burst thread
    t_burst = threading.Thread(target=burst_thread, args=(meta, sof, mcs_walsh, pilot_sym, 2.0), daemon=True)
    t_burst.start()

    print("Streaming continuous channel with bursts in real-time. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

# ---------------- CONFIG -----------------
if __name__ == "__main__":
    meta = {
        'NUM_PACKETS': 10,
        'SNR_VALUES': [1.0, 3.0, 6.0, 9.0, 12.0],
        'N_LDPC': 16200,
        'Rs': 1000,
        'Fs': 4000
    }

    start_streaming(meta)
