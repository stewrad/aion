''' 
Ingest a generated payload packet and segment it for FEC encoding
'''
import numpy as np
import packet_generator as pg

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
    DVB_S2_SHORT_MODCODS = [
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


    modcod_table = {f"{mod}_{rate}": 16200 for mod, rate, idx in DVB_S2_SHORT_MODCODS}

    # Example MCS sequence for 5 frames
    mcs_sequence = ["QPSK_1/2", "8PSK_3/4", "16APSK_2/3", "QPSK_1/2", "32APSK_5/6"]

    frames, pad_lengths = pg.ldpc_bit_segmentation(bits, mcs_sequence, modcod_table)

    print(f"\n   Total frames: {len(frames)}")
    for i, (f, p) in enumerate(zip(frames, pad_lengths)):
        print(f"   Frame {i:02d}: {len(f)} bits, padding added: {p}")