"""
Continuous RF Transmission with DVB-S2 Bursts interspersed

This script performs:
- packet_generator.py (packet and bitstream generation)
- fec_segmentation.py (MCS-based bit segmentation for FEC)
- enc_dec.py (DVB-S2 standard LDPC encoding on short frame (N=16200))
- modem.py (Modulation)
- ZMQ IQ streaming to GNU Radio 
"""
import logging
import threading
import time
import zmq
import numpy as np
import scipy.sparse as sp
from sionna.phy.fec import utils
from enc_dec import compute_generator_matrix
from fec_segmentation import segment_bits, generate_bitstream_from_packets
from plheader import sof_gen, mcs_walsh_gen, pilot_gen, plh_mod
from channel import awgn
from modems.apsk import get_dvbs2_apsk_radii

from mcs_config import MCS_TABLE, MCS_LOOKUP, ALIST_MAP, MOD_MAP, select_mcs_for_snr, insert_pilots

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

def build_burst(snr, meta, sof, mcs_walsh, pilot_sym):
    """
    Generate and encode one DVB-S2 burst (using your pipeline)
    """
    num_packets = meta['PACKETS_PER_BURST']
    Rs = meta['Rs']
    Fs = meta['Fs']

    mcs_name = select_mcs_for_snr(snr)
    mcs_entry = MCS_LOOKUP[mcs_name]
    code_rate = mcs_entry['code_rate']
    mod_type = mcs_entry['mod']
    mcs_idx = mcs_entry['idx']

    logging.info(f"Selected MCS: {mcs_name} ({mod_type}, R={code_rate:.3f} @ SNR {snr:.01f} dB)")
    
    bitstream = generate_bitstream_from_packets(num_packets)
    segments, meta2 = segment_bits(bitstream, [mcs_name])
    logging.info(f"Total bitstream length: {len(bitstream)} bits")

    # Load LDPC matrix
    ALIST_FILE = ALIST_MAP[code_rate]
    with suppress_output():
        alist = utils.load_alist(ALIST_FILE)
        H_dense, _, N, _ = utils.alist2mat(alist)
        M = H_dense.shape[0]
        K = N - M
        H_sparse = sp.csr_matrix(H_dense, dtype=np.uint8)
        G = compute_generator_matrix(H_dense)

    encoded_frames = []
    for frame_bits in segments:
        input_bits = frame_bits[:K] if len(frame_bits) >= K else np.concatenate(
            [frame_bits, np.zeros(K - len(frame_bits), dtype=np.uint8)]
        )
        codeword = (input_bits @ G % 2).astype(np.uint8)
        encoded_frames.append(codeword)

    logging.info(f"Encoded {len(encoded_frames)} codewords (total {len(encoded_frames) * N} bits).")

    mod_func = MOD_MAP[mod_type]
    if mod_type in ['16APSK', '32APSK']:
        radii = get_dvbs2_apsk_radii(mod_type, code_rate)
        if mod_type == '16APSK':
            _, bb, _, _, _ = mod_func(encoded_frames[0], symbol_rate=Rs, sample_rate=Fs, r1=radii[0], r2=radii[1])
        else:
            _, bb, _, _, _ = mod_func(encoded_frames[0], symbol_rate=Rs, sample_rate=Fs, r1=radii[0], r2=radii[1], r3=radii[2])
    else: 
        _, bb, _, _, _ = mod_func(encoded_frames[0], symbol_rate=Rs, sample_rate=Fs)
    logging.info(f'Modulated {len(bb)} Symbols as {mod_type}')

    bb_w_pilot = insert_pilots(bb, pilot_sym)
    _, mcs_sym, _, _, _ = plh_mod(mcs_walsh[:, mcs_idx], Rs, Fs)
    symbol_frame = np.concatenate([sof, mcs_sym, bb_w_pilot])

    # Apply AWGN
    noisy_frame = awgn.awgn_gen(symbol_frame, snr)
    return noisy_frame.astype(np.complex64)

# --------------------------
# Noise Thread
# --------------------------
def noise_thread(socket, Fs: float=4000, noise_level: float=0.05):
    frame_len = int(Fs / 10)
    while True:
        noise = (
            np.random.normal(0, noise_level, frame_len)
            + 1j * np.random.normal(0, noise_level, frame_len)
        ).astype(np.complex64)
        socket.send(noise.tobytes())
        time.sleep(frame_len / Fs)

# --------------------------
# Burst Thread
# --------------------------
def burst_thread(socket, meta, sof, mcs_walsh, pilot_sym, burst_active):
    num_bursts = meta['NUM_BURSTS']
    # for pkt_idx in range(num_bursts):
    #     snr = np.random.uniform(30, 60)
    #     burst = build_burst(snr, meta, sof, mcs_walsh, pilot_sym)
    #     socket.send(burst.tobytes())
    #     logging.info(f"[TX] Sent burst {pkt_idx+1}/{num_bursts} (SNR={snr:.1f} dB)")
    #     logging.info("=============================================================")
    #     time.sleep(meta['burst_interval'])

    i = 0
    while True:
        if burst_active.is_set():
            i += 1
            snr = np.random.uniform(3, 10)
            burst = build_burst(snr, meta, sof, mcs_walsh, pilot_sym)
            socket.send(burst.tobytes())
            logging.info(f"[TX] Sent burst {i} (SNR={snr:.1f} dB)")
            time.sleep(meta['burst_interval'])
        else:
            time.sleep(0.1)

    logging.info(f"[TX] All {num_bursts} bursts transmitted.")

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
    os.system('clear')

    logging.info(f"====== Initializing ACM TX Simulation ======")

    # Configuration: 
    meta = {
        'PACKETS_PER_BURST': 10,
        'NUM_BURSTS': 15, 
        'N_LDPC': 16200, 
        'Rs': 1000,
        'Fs': 4000,
        'burst_interval': 1.0,
        'noise_level': 0.02,
        'burst_amp': 0.5,
        'stream_addr': 'tcp://0.0.0.0:5555'
        }

    # Start of Frame generation and modulation to PI/2-BPSK
    sof_seq = sof_gen()
    _, sof, _, _, _ = plh_mod(sof_seq, meta['Rs'], meta['Fs'])

    # MCS Indices generated as Walsh Codes 
    mcs_walsh = mcs_walsh_gen(32)

    # Pilot Symbol Generation and modulation as PI/2-BPSK
    pilots = pilot_gen()
    _, pilot_sym, _, _, _ = plh_mod(pilots, meta['Rs'], meta['Fs'])
    # print(f'Shape Pilots: {np.shape(pilot_sym)}')

    Fs = meta['Fs']
    Rs = meta['Rs']
    # num_packets = meta['NUM_PACKETS']
    burst_interval = meta.get('burst_interval', 1.0)
    noise_level = meta.get('noise_level', 0.02)
    burst_amp = meta.get('burst_amp', 0.5)
    N_LDPC = meta['N_LDPC']

    # Setting ZMQ Stream 
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind("tcp://0.0.0.0:5555")
    logging.info("=============================================================")
    logging.info(f"[TX] Bound to tcp://0.0.0.0:5555")

    # --------------------------
    # Start Threads
    # --------------------------
    burst_active = threading.Event()

    t_noise = threading.Thread(target=noise_thread, args=(socket, Fs, noise_level), daemon=True)
    t_burst = threading.Thread(target=burst_thread, args=(socket, meta, sof, mcs_walsh, pilot_sym, burst_active), daemon=True)
    t_noise.start()
    t_burst.start()


    print("\nCommands: 'on' = start bursts | 'off' = stop bursts | 'quit' = exit\n")
    try:
        while True:
            cmd = input("TX> ").strip().lower()
            if cmd == "on":
                burst_active.set()
                print("[TX] Bursts enabled.")
            elif cmd == "off":
                burst_active.clear()
                print("[TX] Bursts paused.")
            elif cmd == "quit":
                print("[TX] Exiting...")
                break
            else:
                print("Commands: on / off / quit")
    except KeyboardInterrupt:
        pass

    # try:
    #     while t_burst.is_alive():
    #         time.sleep(0.5)
    # except KeyboardInterrupt:
    #     logging.info("[TX] Interrupted by user.")


# Generate simulated data RF files and simulated human annotated tags
if __name__ == "__main__":
    main()

