''' 
Ingest a generated payload packet and segment it for FEC encoding
'''
import numpy as np
import packet_generator as pg

from gnuradio import fec
import math
import sys
import traceback

def segment_bits_for_ldpc(bits, K):
    segments = []
    while len(bits) > K:
        segments.append(bits[:K])
        bits = bits[K:]
    if len(bits) > 0:
        # pad or shorten last segment
        pad_len = K - len(bits)
        bits += [0] * pad_len
        segments.append(bits)
    return segments

def parse_rate(rate_str):
    """Parse 'num/den' string to float."""
    num, den = rate_str.split('/')
    return float(num) / float(den)

def expected_coded_len(K, rate_str):
    """Return expected LDPC encoded length N for given input K and rate string like '3/4'."""
    R = parse_rate(rate_str)
    return int(round(K / R))


def parse_rate_frac(rate_str):
    """Return float rate and (num,den) for a string 'num/den'."""
    num, den = rate_str.split('/')
    return float(num) / float(den), int(num), int(den)

def encode_bits_by_mcs(bits, mcs_sequence, verbose=True):
    """
    Segment `bits` into per-frame K blocks determined by each MCS in `mcs_sequence`
    (sequence can be shorter than required; it will be repeated), then LDPC-encode
    each block using DVB-S2 short-frame matrices via gr-fec.

    Parameters
    ----------
    bits : 1D np.ndarray dtype {0,1}
    mcs_sequence : list of strings like "QPSK_3/4" or "8PSK_2/3"
    verbose : bool

    Returns
    -------
    encoded_frames : list of np.ndarray (each length N_SHORT)
    frame_infos : list of dict with keys: idx, mcs, mod, rate_str, K, N, pad_len
    """
    N_SHORT = 16200

    if not isinstance(bits, np.ndarray):
        bits = np.array(bits, dtype=int)
    bits = bits.astype(np.uint8)
    total_bits = len(bits)

    if verbose:
        print(f"[encode_bits_by_mcs] total input bits = {total_bits}")

    if len(mcs_sequence) == 0:
        raise ValueError("mcs_sequence must contain at least one MCS entry")

    encoded_frames = []
    frame_infos = []
    cursor = 0
    frame_idx = 0
    seq_len = len(mcs_sequence)

    while cursor < total_bits:
        mcs_key = mcs_sequence[frame_idx % seq_len]
        if mcs_key not in MODCOD_LOOKUP:
            raise ValueError(f"MCS '{mcs_key}' not found in MODCOD_TABLE")

        mod, rate_str = MODCOD_LOOKUP[mcs_key]
        if rate_str not in RATE_ENUM_MAP:
            raise ValueError(f"Rate '{rate_str}' not supported by RATE_ENUM_MAP")

        # compute K from rate and fixed N
        rate_float, num, den = parse_rate_frac(rate_str)
        K = int(round(N_SHORT * rate_float))
        N = N_SHORT
        pad_len = 0

        # take K bits (pad with zeros if insufficient remaining)
        remaining = total_bits - cursor
        if remaining >= K:
            bits_in = bits[cursor:cursor+K].astype(np.uint8)
        else:
            pad_len = K - remaining
            if verbose:
                print(f"  [frame {frame_idx}] Not enough bits (need {K}, have {remaining}). Padding {pad_len} zeros.")
            bits_in = np.concatenate([bits[cursor:], np.zeros(pad_len, dtype=np.uint8)])

        # instantiate DVB-S2 short LDPC encoder via gr-fec
        rate_enum = RATE_ENUM_MAP[rate_str]
        params = fec.ldpc_code_params.make_dvbs2(N, rate_enum)   # uses built-in DVB-S2 matrices
        encoder = fec.ldpc_encoder_make(params)

        # encode: some GNURadio encoders take (in, out) or return list. Use encoder.encode(in_list)
        in_list = bits_in.astype(int).tolist()
        try:
            out_list = encoder.encode(in_list)   # returns list of 0/1
        except TypeError:
            # fallback: some API variants expect (in_bytes, out_buffer) - try other signature
            out_buf = [0] * N
            encoder.encode(in_list, out_buf)  # may modify out_buf in place
            out_list = out_buf

        encoded = np.array(out_list, dtype=np.uint8)
        if len(encoded) != N:
            raise RuntimeError(f"Encoded length mismatch: got {len(encoded)} expected {N}")

        encoded_frames.append(encoded)
        info = {
            "idx": frame_idx,
            "mcs": mcs_key,
            "mod": mod,
            "rate_str": rate_str,
            "K": K,
            "N": N,
            "pad_len": pad_len,
            "cursor_start": cursor,
            "cursor_end": cursor + min(remaining, K) - 1,
        }
        frame_infos.append(info)

        if verbose:
            print(f"  frame {frame_idx:03d}: MCS={mcs_key:<10} K={K:5d} N={N:5d} pad={pad_len} "
                  f"cursor=[{info['cursor_start']},{info['cursor_end']}]")

        cursor += K
        frame_idx += 1

    if verbose:
        total_encoded = len(encoded_frames) * N_SHORT
        print(f"[encode_bits_by_mcs] total frames encoded = {len(encoded_frames)}, total encoded bits = {total_encoded}")
    return encoded_frames, frame_infos

if __name__ == "__main__":
    print("=== ACM Packet Generator Demo ===\n")

    gen = pg.PacketGenerator(max_payload_size=1276, seed=42)

    # Generate single packets
    print("[1] Generating individual packets:")
    for pkt_type in pg.PacketType:
        packet, metadata = gen.generate_packet(packet_type=pkt_type)
        print(f"   {pkt_type.name}: {len(packet)} bytes, "
              f"seq={metadata.seq_num}, payload={metadata.payload_size} bytes")

        print(type)
    # Generate stream
    print("\n[2] Generating packet stream (100 packets):")
    N = 50
    stream = gen.generate_packet_stream(N)

    type_counts = {}
    total_bytes = 0
    for packet, metadata in stream:
        type_counts[metadata.packet_type.name] = type_counts.get(metadata.packet_type.name, 0) + 1
        total_bytes += len(packet)
    
    print(f"   Total bytes: {total_bytes}")
    print(f"   Traffic distribution: {type_counts}")
    print(f"   Average packet size: {total_bytes / len(stream):.1f} bytes")       

    # print("\3. Writing packet stream to PCAP:")
    ipudp_payloads = [pg.build_ip_udp_wrapper(pkt_bytes) for pkt_bytes, _ in stream]   

    # # Convert to bits for FEC
    # print("\3. Converting to bit array for FEC encoding:")
    # bits = np.concatenate([gen.to_bit_array(p) for p in ipudp_payloads])

    # for k, b in enumerate(bits):
    #     if k <= 10:
    #         print(f'Packet {k}: {(b.)} bits')

    # K_LDPC = 16200 # short frame 
    # ldpc_blocks, pad_len = pg.pad_or_truncate_to_block(bits, K_LDPC)
    # # encoded_blocks = [ldpc_encode(block) for block in ldpc_blocks]

    bits = np.array([], dtype=int)
    packet_bit_lengths = []  # for tracking
    packet_offsets = []      # cumulative starting index per packet

    print("\n[3] Converting IP/UDP payloads to bit arrays for FEC encoding:\n")

    bit_cursor = 0  # keeps track of position in concatenated bitstream

    for i, ip_packet in enumerate(ipudp_payloads):
        bit_array = gen.to_bit_array(ip_packet)
        bits = np.append(bits, bit_array)

        n_bits = len(bit_array)
        packet_bit_lengths.append(n_bits)
        packet_offsets.append(bit_cursor)

        print(f"   Packet {i:02d}: {len(ip_packet):4d} bytes → {n_bits:6d} bits "
            f"(start={bit_cursor:6d}, end={bit_cursor + n_bits - 1:6d})")

        bit_cursor += n_bits

    print(f"\n   Total combined bits: {len(bits)}")




    print("\n[4] Converting combined bits into K bit blocks for LDPC encoding:\n")
    K_LDPC = 16200 # short frame 
    # K_LDPC = 64800 # long frame 
    # ldpc_blocks, pad_len = pg.pad_or_truncate_to_block(bits, K_LDPC)
    # encoded_blocks = [ldpc_encode(block) for block in ldpc_blocks]

    # Define MCS table (DVB-S2-like)
    # Long frames 
    # mcs_table = {
    #     "BPSK_1_2": 32400,
    #     "QPSK_1_2": 32400,
    #     "8PSK_3_4": 48600,
    #     "16APSK_2_3": 43200,
    #     "32APSK_5_6": 54000,
    # }
    
    # Short frames
    # ============= When building ACM sim, map Eb/N0 thresholds to mcs indices =========== 
    MODCOD_TABLE = [
        # (mod, rate, index)
        ("QPSK", "1/4", 1),
        ("QPSK", "1/3", 2),
        ("QPSK", "2/5", 3),
        ("QPSK", "1/2", 4),
        ("QPSK", "3/5", 5),
        ("QPSK", "2/3", 6),
        ("QPSK", "3/4", 7),
        ("QPSK", "4/5", 8),
        ("QPSK", "5/6", 9),
        ("QPSK", "8/9", 10),
        ("QPSK", "9/10", 11),
        ("8PSK", "3/5", 12),
        ("8PSK", "2/3", 13),
        ("8PSK", "3/4", 14),
        ("8PSK", "5/6", 15),
        ("8PSK", "8/9", 16),
        ("8PSK", "9/10", 17),
        ("16APSK", "2/3", 18),
        ("16APSK", "3/4", 19),
        ("16APSK", "4/5", 20),
        ("16APSK", "5/6", 21),
        ("16APSK", "8/9", 22),
        ("16APSK", "9/10", 23),
        ("32APSK", "3/4", 24),
        ("32APSK", "4/5", 25),
        ("32APSK", "5/6", 26),
        ("32APSK", "8/9", 27),
        ("32APSK", "9/10", 28),
    ]


    # Build a lookup: "MOD_RATE" -> (mod, rate_str)
    MODCOD_LOOKUP = { f"{mod}_{rate}": (mod, rate) for (mod, rate, _) in MODCOD_TABLE }

    # Map rate string to fec.ldpc_code_params enum names
    # Note: these attribute names are present in gnuradio.fec.ldpc_code_params
    RATE_ENUM_MAP = {
        "1/4":  fec.ldpc_code_params.rate_1_4,
        "1/3":  fec.ldpc_code_params.rate_1_3,
        "2/5":  fec.ldpc_code_params.rate_2_5,
        "1/2":  fec.ldpc_code_params.rate_1_2,
        "3/5":  fec.ldpc_code_params.rate_3_5,
        "2/3":  fec.ldpc_code_params.rate_2_3,
        "3/4":  fec.ldpc_code_params.rate_3_4,
        "4/5":  fec.ldpc_code_params.rate_4_5,
        "5/6":  fec.ldpc_code_params.rate_5_6,
        "8/9":  fec.ldpc_code_params.rate_8_9,
        "9/10": fec.ldpc_code_params.rate_9_10,
    }

    # # Map to LDPC K values (all short frames: K = 16200)
    # mcs_table = {f"{mod}_{rate}": 16200 for mod, rate, idx in DVB_S2_SHORT_MODCODS}

    # # Example test MCS sequence for 5 frames
    # mcs_sequence = [
    #     "QPSK_1/2",
    #     "8PSK_3/4",
    #     "16APSK_2/3",
    #     "QPSK_1/2",
    #     "32APSK_5/6",
    # ]

    # frames = pg.segment_into_frames(bits, mcs_sequence, mcs_table)
    
    # print(f"\nGenerated {len(frames)} frames for testing.")
    # for i, f in enumerate(frames):
    #     print(f"Frame {i:02d}: {len(f)} bits")


    # modcod_table = {f"{mod}_{rate}": 16200 for mod, rate, idx in DVB_S2_SHORT_MODCODS}

    # Example MCS sequence for 5 frames
    mcs_sequence = ["QPSK_1/2", "8PSK_3/4", "16APSK_2/3", "QPSK_1/2", "32APSK_5/6"]

    frames, pad_lengths = pg.ldpc_bit_segmentation(bits, mcs_sequence, modcod_table)

    print(f"\n   Total frames: {len(frames)}")
    for i, (f, p) in enumerate(zip(frames, pad_lengths)):
        print(f"   Frame {i:02d}: {len(f)} bits, padding added: {p}")


    print(f"\n[5] Encoding {len(frames)} with LDPC:\n")  

    # Example MCS sequence (will be repeated to exhaust bits)
    mcs_sequence = ["QPSK_1/2", "8PSK_3/4", "16APSK_2/3"]  # use keys matching MODCOD_TABLE entries

    encoded_frames, frame_infos = encode_bits_by_mcs(bits, mcs_sequence, verbose=True)

    # quick checks
    print(f"Encoded frames: {len(encoded_frames)}")
    print("First frame info:", frame_infos[0])
    print("First encoded frame length:", len(encoded_frames[0]))