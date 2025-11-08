import numpy as np
from typing import List, Tuple, Dict
import packet_generator as pg

# ================================
# Define your MCS table
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

MCS_LOOKUP = {entry['name']: entry for entry in MCS_TABLE}

# Fixed LDPC frame length
N_LDPC = 16200

# ================================
# Segment bits based on MCS
# ================================
def segment_bits(bits: np.ndarray, mcs_sequence: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
    if not isinstance(bits, np.ndarray):
        bits = np.array(bits, dtype=int)
    bits = bits.astype(np.uint8)

    segments = []
    metadata = []
    cursor = 0
    frame_idx = 0
    seq_len = len(mcs_sequence)
    total_bits = len(bits)

    while cursor < total_bits:
        mcs_name = mcs_sequence[frame_idx % seq_len]
        if mcs_name not in MCS_LOOKUP:
            raise ValueError(f"MCS '{mcs_name}' not found in MCS_TABLE")
        
        mcs = MCS_LOOKUP[mcs_name]
        K = int(round(N_LDPC * mcs['code_rate']))

        remaining = total_bits - cursor
        pad_len = 0
        if remaining >= K:
            segment = bits[cursor:cursor+K]
        else:
            pad_len = K - remaining
            segment = np.concatenate([bits[cursor:], np.zeros(pad_len, dtype=np.uint8)])
        
        segments.append(segment)
        metadata.append({
            'frame_idx': frame_idx,
            'mcs': mcs_name,
            'mod': mcs['mod'],
            'code_rate': mcs['code_rate'],
            'K': K,
            'N': N_LDPC,
            'pad_len': pad_len,
            'cursor_start': cursor,
            'cursor_end': cursor + min(K, remaining) - 1,
        })
        
        cursor += K
        frame_idx += 1

    return segments, metadata

# ================================
# Generate packet stream and convert to bits
# ================================
def generate_bitstream_from_packets(num_packets: int, seed: int = 42) -> np.ndarray:
    gen = pg.PacketGenerator(max_payload_size=1276, seed=seed)
    stream = gen.generate_packet_stream(num_packets)

    bits_list = []
    for packet, _ in stream:
        bits_list.append(gen.to_bit_array(packet))

    return np.concatenate(bits_list).astype(np.uint8)

# ================================
# Example usage
# ================================
if __name__ == "__main__":
    NUM_PACKETS = 150
    MCS_SEQUENCE = ["QPSK-1/2", "8APSK-3/4", "16APSK-2/3", "QPSK-3/4"]  # repeats automatically

    # Generate bitstream from packets
    bits = generate_bitstream_from_packets(NUM_PACKETS)
    print(f"Generated {len(bits)} bits from {NUM_PACKETS} packets.\n")

    # Segment bits according to MCS
    segments, meta = segment_bits(bits, MCS_SEQUENCE)

    print(f"Segmented into {len(segments)} K-bit blocks (LDPC ready).")
    print("First 10 frames (metadata):")
    for i in range(min(10, len(segments))):
        m = meta[i]
        print(f"Frame {i}: MCS={m['mcs']}, Mod={m['mod']}, K={m['K']}, N={m['N']}, Pad={m['pad_len']}")
