import logging
import threading, time, zmq, json, numpy as np
import scipy.sparse as sp
from sionna.phy.fec import utils
from enc_dec import compute_generator_matrix
from fec_segmentation import segment_bits, generate_bitstream_from_packets
from plheader import sof_gen, mcs_walsh_gen, pilot_gen, plh_mod
from modems import psk
from channel import awgn
from mcs_config import MCS_LOOKUP, ALIST_MAP, MOD_MAP, insert_pilots

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


# =====================================================
# --- Global runtime state ---
# =====================================================
current_mcs_idx = 0     # updated by receiver
burst_active = True     # can be toggled on/off
Fs = 4000
noise_level = 0.02
burst_interval = 1.0

# =====================================================
# --- DVB-S2 burst builder (same as before) ---
# =====================================================
def build_burst(mcs_idx, meta, sof, mcs_walsh, pilots):
    """Build and modulate a single DVB-S2 burst using the selected MCS."""
    snr = np.random.uniform(4, 9)
    Rs = meta['Rs']
    Fs = meta['Fs']
    sps = int(round(Fs/Rs))
    num_bursts = meta['NUM_BURSTS']

    # lookup by index
    mcs_entry = list(MCS_LOOKUP.values())[mcs_idx % len(MCS_LOOKUP)]
    code_rate = mcs_entry['code_rate']
    mod_type = mcs_entry['mod']
    name = mcs_entry['name']
    logging.info(f"Selected MCS: {name} ({mod_type}, R={code_rate:.3f} @ SNR {snr:.01f} dB)")

    # generate bitstream & segment
    bitstream = generate_bitstream_from_packets(num_bursts)
    segments, _ = segment_bits(bitstream, [name])
    logging.info(f"Total bitstream length: {len(bitstream)} bits")

    # LDPC encode
    ALIST_FILE = ALIST_MAP[code_rate]
    with suppress_output():
        alist = utils.load_alist(ALIST_FILE)
        H_dense, _, N, _ = utils.alist2mat(alist)
        M = H_dense.shape[0]
        K = N - M
        G = compute_generator_matrix(H_dense)

    # frame_bits = segments[0]
    # input_bits = frame_bits[:K] if len(frame_bits) >= K else np.concatenate(
    #     [frame_bits, np.zeros(K - len(frame_bits), dtype=np.uint8)]
    # )
    # codeword = (input_bits @ G % 2).astype(np.uint8)

    encoded_frames = []
    for frame_bits in segments:
        input_bits = frame_bits[:K] if len(frame_bits) >= K else np.concatenate(
            [frame_bits, np.zeros(K - len(frame_bits), dtype=np.uint8)]
        )
        codeword = (input_bits @ G % 2).astype(np.uint8)
        encoded_frames.append(codeword)

    logging.info(f"Encoded {len(encoded_frames)} codewords (total {len(encoded_frames) * N} bits).")


    # # modulation
    # mod_func = MOD_MAP[mod_type]
    # _, bb, _, _, _ = mod_func(codeword, symbol_rate=Rs, sample_rate=Fs)

    # # insert pilots and PL headers
    # bb_w_pilot = insert_pilots(bb, pilots, samples_per_symbol=sps)
    # _, mcs_sym, _, _, _ = plh_mod(mcs_walsh[:, mcs_idx], Rs, Fs)
    # frame = np.concatenate([sof, mcs_sym, bb_w_pilot])

    # # add AWGN
    # noisy = awgn.awgn_gen(frame, snr)
    # return noisy.astype(np.complex64)

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

    logging.info(f'Modulated {len(bb)//sps} symbols ({len(bb)} samples) as {mod_type}')
    logging.info(f'Pilot block: {len(pilots)} samples')

    # Insert pilots at DVB-S2 symbol intervals
    bb_w_pilot = insert_pilots(
        data_samples=bb,
        pilot_samples=pilots,
        samples_per_symbol=sps,
        data_symbols=1440,      # DVB-S2 standard
        pilot_symbols=36        # DVB-S2 standard
    )
    _, mcs_samples, _, _, _ = plh_mod(mcs_walsh[:, mcs_idx], Rs, Fs)
    sample_frame = np.concatenate([sof, mcs_samples, bb_w_pilot])

    # Apply AWGN
    noisy_frame = awgn.awgn_gen(sample_frame, snr)
    return noisy_frame.astype(np.complex64)

# =====================================================
# --- ZMQ sockets ---
# =====================================================
ctx = zmq.Context()
tx_sock = ctx.socket(zmq.PUB)
tx_sock.bind("tcp://0.0.0.0:5555")

rx_sock = ctx.socket(zmq.SUB)
rx_sock.connect("tcp://127.0.0.1:5556")
rx_sock.setsockopt_string(zmq.SUBSCRIBE, "")

# =====================================================
# --- Threads ---
# =====================================================
def csi_listener():
    global current_mcs_idx
    while True:
        try:
            msg = rx_sock.recv_json(flags=0)
            if "recommended_mcs_idx" in msg:
                current_mcs_idx = msg["recommended_mcs_idx"]
                logging.info(f"[CSI] Updated MCS index → {current_mcs_idx}")
        except Exception:
            time.sleep(0.1)

def noise_streamer():
    """Continuous channel noise stream"""
    frame_len = int(Fs / 10)
    while True:
        noise = (np.random.randn(frame_len) + 1j*np.random.randn(frame_len)) * noise_level
        tx_sock.send(noise.astype(np.complex64).tobytes())
        time.sleep(frame_len / Fs)

def burst_streamer(meta, sof, mcs_walsh, pilots):
    global current_mcs_idx, burst_active
    pkt = 0
    while True:
        if not burst_active:
            time.sleep(0.1)
            continue
        burst = build_burst(current_mcs_idx, meta, sof, mcs_walsh, pilots)
        tx_sock.send(burst.tobytes())
        pkt += 1
        logging.info(f"[TX] Burst #{pkt} sent (MCS={current_mcs_idx})")
        time.sleep(burst_interval)

# Set up basic configuration (can be made configurable)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for verbose output
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)    # for full debug output
logger.setLevel(logging.WARNING)  # for quieter output

# =====================================================
# --- Main ---
# =====================================================
def main():
    os.system('clear')
    logging.info(f"====== Initializing ACM TX Simulation ======")
    logging.info("=============================================================")
    logging.info(f"[TX] Bound to tcp://0.0.0.0:5555")
    logging.info(f"[TX] Connected to CSI feedback at tcp://127.0.0.1:5556")
    
    meta = {
        'NUM_BURSTS': 10,
        'N_LDPC': 16200,
        'Rs': 1000,
        'Fs': 4000
    }

    sof_seq = sof_gen()
    _, sof, _, _, _ = plh_mod(sof_seq, meta['Rs'], meta['Fs'])
    mcs_walsh = mcs_walsh_gen(32)
    pilot_bits = pilot_gen()
    _, pilots, _, _, _ = plh_mod(pilot_bits, meta['Rs'], meta['Fs'])

    threading.Thread(target=noise_streamer, daemon=True).start()
    threading.Thread(target=burst_streamer, args=(meta, sof, mcs_walsh, pilots), daemon=True).start()
    threading.Thread(target=csi_listener, daemon=True).start()

    logging.info("\nCommands: 'on' = start bursts | 'off' = stop bursts | 'quit' = exit\n")
    try:
        while True:
            cmd = input("TX> ").strip().lower()
            if cmd == "on":
                burst_active = True
                logging.info("[TX] Bursts enabled.")
            elif cmd == "off":
                burst_active = False
                logging.info("[TX] Bursts paused.")
            elif cmd == "quit":
                logging.info("[TX] Exiting...")
                break
            else:
                logging.info("Commands: on / off / quit")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()