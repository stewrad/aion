import numpy as np
import struct
import time
import binascii
import re
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple
from scapy.all import Ether, IP, UDP, Raw, wrpcap, rdpcap


# ---------- Packet definitions ----------
class PacketType(IntEnum):
    """Packet types for different traffic classes"""
    DATA = 0
    VOICE = 1
    VIDEO = 2
    CONTROL = 3


@dataclass
class PacketMetadata:
    """Metadata for tracking packet information"""
    seq_num: int
    timestamp: float
    packet_type: PacketType
    payload_size: int
    priority: int


# ---------- Packet Generator ----------
class PacketGenerator:
    """Realistic packet generator for ACM communication link simulation."""

    def __init__(self, max_payload_size: int = 1500, seed: int = None):
        self.max_payload_size = max_payload_size
        self.seq_num = 0
        if seed is not None:
            np.random.seed(seed)

        self.header_size = 16
        self.crc_size = 4
        self.min_packet_size = self.header_size + self.crc_size

    def _compute_crc32(self, data: bytes) -> int:
        """Compute CRC32 checksum for error detection"""
        return np.uint32(np.bitwise_xor.reduce(
            np.frombuffer(data, dtype=np.uint8).view(np.uint32)
        )) if len(data) % 4 == 0 else hash(data) & 0xFFFFFFFF

    def _create_header(self, packet_type: PacketType, payload_size: int) -> bytes:
        version = 1
        header = struct.pack('!BBIdd', version, packet_type, self.seq_num, time.time(), payload_size)
        self.seq_num += 1
        return header

    def _generate_realistic_payload(self, size: int, packet_type: PacketType) -> bytes:
        if packet_type == PacketType.DATA:
            payload = np.random.bytes(size)
        elif packet_type == PacketType.VOICE:
            frame_pattern = np.random.randint(0, 128, size=size // 2, dtype=np.uint8)
            payload = np.repeat(frame_pattern, 2).tobytes()[:size]
        elif packet_type == PacketType.VIDEO:
            if np.random.random() < 0.1:
                payload = np.random.bytes(size)
            else:
                base = np.random.randint(0, 256, size=size // 4, dtype=np.uint8)
                payload = np.repeat(base, 4).tobytes()[:size]
        elif packet_type == PacketType.CONTROL:
            pattern = np.array([0xAA, 0x55] * (size // 2), dtype=np.uint8)
            payload = pattern.tobytes()[:size]
        else:
            payload = np.random.bytes(size)
        return payload

    def generate_packet(self, payload_size: int = None, packet_type: PacketType = PacketType.DATA) -> Tuple[bytes, PacketMetadata]:
        if payload_size is None:
            if packet_type == PacketType.VOICE:
                payload_size = np.random.choice([160, 320, 640])
            elif packet_type == PacketType.VIDEO:
                payload_size = int(np.random.exponential(800))
                payload_size = min(payload_size, self.max_payload_size)
            elif packet_type == PacketType.CONTROL:
                payload_size = np.random.randint(64, 256)
            else:
                payload_size = np.random.randint(64, self.max_payload_size)
        payload_size = min(payload_size, self.max_payload_size)

        header = self._create_header(packet_type, payload_size)
        payload = self._generate_realistic_payload(payload_size, packet_type)
        packet_without_crc = header + payload
        crc = self._compute_crc32(packet_without_crc)
        crc_bytes = struct.pack('!I', crc)
        complete_packet = packet_without_crc + crc_bytes

        metadata = PacketMetadata(
            seq_num=self.seq_num - 1,
            timestamp=time.time(),
            packet_type=packet_type,
            payload_size=payload_size,
            priority=int(packet_type)
        )

        return complete_packet, metadata

    def generate_packet_stream(self, num_packets: int, traffic_mix: dict = None) -> List[Tuple[bytes, PacketMetadata]]:
        if traffic_mix is None:
            traffic_mix = {
                PacketType.DATA: 0.6,
                PacketType.VOICE: 0.15,
                PacketType.VIDEO: 0.2,
                PacketType.CONTROL: 0.05
            }
        total = sum(traffic_mix.values())
        traffic_probs = [traffic_mix.get(PacketType(i), 0) / total for i in range(4)]

        packets = []
        for _ in range(num_packets):
            packet_type = PacketType(np.random.choice(4, p=traffic_probs))
            packet, metadata = self.generate_packet(packet_type=packet_type)
            packets.append((packet, metadata))

        return packets

    def verify_packet(self, packet: bytes) -> Tuple[bool, dict]:
        if len(packet) < self.min_packet_size:
            return False, {"error": "Packet too short"}
        try:
            header = packet[:self.header_size]
            crc_received = struct.unpack('!I', packet[-self.crc_size:])[0]
            packet_without_crc = packet[:-self.crc_size]
            crc_computed = self._compute_crc32(packet_without_crc)
            is_valid = (crc_received == crc_computed)
            version, pkt_type, seq, timestamp, length = struct.unpack('!BBIdd', header)
            parsed_info = {
                "valid": is_valid,
                "version": version,
                "type": PacketType(pkt_type).name,
                "sequence": seq,
                "timestamp": timestamp,
                "payload_length": int(length),
                "total_length": len(packet)
            }
            return is_valid, parsed_info
        except Exception as e:
            return False, {"error": str(e)}

    def to_bit_array(self, packet: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(packet, dtype=np.uint8))

    def from_bit_array(self, bits: np.ndarray) -> bytes:
        remainder = len(bits) % 8
        if remainder != 0:
            bits = np.append(bits, np.zeros(8 - remainder, dtype=np.uint8))
        return np.packbits(bits).tobytes()


# ---------- Ethernet/IP/UDP helpers ----------
def build_ip_udp_wrapper(inner_payload: bytes, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=5000, dst_port=6000) -> bytes:
    """
    Wrap the ACM packet (with CRC included) inside IP/UDP
    The Raw layer preserves the full ACM bytes for later verification.
    """
    pkt = IP(src=src_ip, dst=dst_ip) / UDP(sport=src_port, dport=dst_port) / Raw(inner_payload)
    return bytes(pkt)


def frames_to_pcap(frames: List[bytes], filename="frames.pcap", src_mac="11:22:33:44:55:66", dst_mac="aa:bb:cc:dd:ee:ff"):
    """Write frames to PCAP with Ethernet headers; Raw includes ACM + CRC bytes."""
    scapy_packets = []
    for payload in frames:
        eth_pkt = Ether(dst=dst_mac, src=src_mac, type=0x0800) / Raw(payload)
        scapy_packets.append(eth_pkt)
    wrpcap(filename, scapy_packets)
    print(f"Saved {len(scapy_packets)} frames to {filename}")

import numpy as np


def pad_or_truncate_to_block(bits: np.ndarray, k_block: int, verbose: bool = True):
    """
    Pad or truncate bit array to fit integer number of LDPC input blocks.

    Parameters
    ----------
    bits : np.ndarray
        Input bit array (0/1).
    k_block : int
        LDPC block size in bits (e.g., 64800 or 16200).
    verbose : bool
        Whether to print debug information.

    Returns
    -------
    blocks : np.ndarray
        Reshaped (num_blocks, k_block) array ready for encoding.
    pad_len : int
        Number of padding bits added.
    """

    # --- Input validation ---
    if not isinstance(bits, np.ndarray):
        raise TypeError("Input bits must be a NumPy array.")
    if bits.ndim != 1:
        raise ValueError("Input bit array must be 1D.")
    if not np.isin(bits, [0, 1]).all():
        raise ValueError("Input bit array must contain only 0 and 1.")
    if not isinstance(k_block, int) or k_block <= 0:
        raise ValueError("LDPC block size (k_block) must be a positive integer.")

    total_bits = len(bits)
    remainder = total_bits % k_block

    # --- Debug info ---
    if verbose:
        print(f"\n[LDPC Block Preparation]")
        print(f"  Total payload bits: {total_bits}")
        print(f"  Target LDPC block size: {k_block}")
        print(f"  Remainder bits: {remainder}")

    # --- Handle padding or truncation ---
    pad_len = 0
    if remainder == 0:
        num_blocks = total_bits // k_block
        if verbose:
            print(f"  ✅ Perfect alignment: {num_blocks} block(s) of {k_block} bits each.")
    elif remainder < k_block // 2:
        pad_len = k_block - remainder
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])
        num_blocks = len(bits) // k_block
        if verbose:
            print(f"  ⚠️ Padding last block with {pad_len} zero bits ({pad_len/8:.2f} bytes).")
    else:
        # Truncate last incomplete block if too large to pad efficiently
        truncate_len = remainder
        bits = bits[:-truncate_len]
        num_blocks = len(bits) // k_block
        if verbose:
            print(f"  ⚠️ Truncated {truncate_len} extra bits to fit exact block size.")

    # --- Sanity check ---
    if len(bits) % k_block != 0:
        raise RuntimeError("Padding/truncation failed — bit array still not aligned.")

    # --- Reshape into blocks ---
    blocks = bits.reshape(num_blocks, k_block)

    if verbose:
        print(f"  🧱 Created {num_blocks} LDPC block(s) of {k_block} bits each.")
        print(f"  Total output bits: {len(bits)}\n")

    return blocks, pad_len


# NEED FOR INTEGRATION WITH MCS 
def segment_into_frames(bits, mcs_sequence, mcs_table):
    """
    Divide bitstream into frames according to MCS-dependent LDPC block sizes.

    Parameters
    ----------
    bits : np.ndarray
        Bit array to segment.
    mcs_sequence : list of str
        Ordered list of MCS identifiers for each frame, e.g. ['QPSK_1_2', '8PSK_3_4', ...]
    mcs_table : dict
        Maps MCS identifier -> LDPC K size (input bits)

    Returns
    -------
    frames : list of np.ndarray
        Each item is a segment of bits of length K_MCS.
    """

    frames = []
    cursor = 0
    total_bits = len(bits)

    for i, mcs in enumerate(mcs_sequence):
        K = mcs_table[mcs]
        if cursor + K <= total_bits:
            frame_bits = bits[cursor : cursor + K]
        else:
            # Last frame — pad with zeros if short
            pad_len = cursor + K - total_bits
            frame_bits = np.concatenate([bits[cursor:], np.zeros(pad_len, dtype=int)])
            print(f"⚠️ Frame {i} padded with {pad_len} bits.")
        frames.append(frame_bits)
        print(f"Frame {i:02d}: {mcs:<10} | {K:6d} bits (start={cursor}, end={cursor+K-1})")
        cursor += K

    return frames

def ldpc_bit_segmentation(bits, mcs_sequence, modcod_table, verbose=True):
    """
    Segment a global bit array into short DVB-S2 frames using MCS sequence.
    Cycles through the mcs_sequence until all bits are consumed.

    Returns frames and padding info.
    """
    frames = []
    pad_lengths = []
    cursor = 0
    total_bits = len(bits)
    seq_len = len(mcs_sequence)
    frame_count = 0

    while cursor < total_bits:
        mcs_name = mcs_sequence[frame_count % seq_len]
        if mcs_name not in modcod_table:
            raise ValueError(f"MCS '{mcs_name}' not found in modcod_table.")
        K = modcod_table[mcs_name]

        # Slice bits
        if cursor + K <= total_bits:
            frame_bits = bits[cursor:cursor+K]
            pad_len = 0
        else:
            pad_len = cursor + K - total_bits
            frame_bits = np.concatenate([bits[cursor:], np.zeros(pad_len, dtype=int)])
            if verbose:
                print(f"⚠️ Frame {frame_count} padded with {pad_len} bits to reach K={K}")

        frames.append(frame_bits)
        pad_lengths.append(pad_len)

        if verbose:
            print(f"   Frame {frame_count:03d}: {mcs_name:<10} | K={K} bits | "
                  f"start={cursor} end={cursor+len(frame_bits)-1} | pad={pad_len}")

        cursor += K
        frame_count += 1

    return frames, pad_lengths

# ---------- Main demo ----------
if __name__ == "__main__":
    print("=== ACM Packet Generator Demo ===\n")

    gen = PacketGenerator(max_payload_size=1500, seed=42)

    # Generate single packets
    print("1. Generating individual packets:")
    for pkt_type in PacketType:
        packet, metadata = gen.generate_packet(packet_type=pkt_type)
        print(f"   {pkt_type.name}: {len(packet)} bytes, "
              f"seq={metadata.seq_num}, payload={metadata.payload_size} bytes")

    # Generate stream
    print("\n2. Generating packet stream (100 packets):")
    stream = gen.generate_packet_stream(100)

    type_counts = {}
    total_bytes = 0
    for packet, metadata in stream:
        type_counts[metadata.packet_type.name] = type_counts.get(metadata.packet_type.name, 0) + 1
        total_bytes += len(packet)

    print(f"   Total bytes: {total_bytes}")
    print(f"   Traffic distribution: {type_counts}")
    print(f"   Average packet size: {total_bytes / len(stream):.1f} bytes")

    # Verify a test packet
    print("\n3. Verifying packet integrity:")
    test_packet, _ = gen.generate_packet(payload_size=500, packet_type=PacketType.DATA)
    is_valid, info = gen.verify_packet(test_packet)
    print(f"   Packet valid: {is_valid}")
    print(f"   Parsed info: {info}")

    # Convert to bits for FEC
    print("\n4. Converting to bit array for FEC encoding:")
    bits = gen.to_bit_array(test_packet)
    print(f"   Packet size: {len(test_packet)} bytes = {len(bits)} bits")

    # Simulate bit errors
    print("\n5. Simulating bit errors (for FEC testing):")
    corrupted_bits = bits.copy()
    error_positions = np.random.choice(len(bits), size=5, replace=False)
    corrupted_bits[error_positions] = 1 - corrupted_bits[error_positions]

    recovered_packet = gen.from_bit_array(corrupted_bits)
    is_valid_after, _ = gen.verify_packet(recovered_packet)
    print(f"   Introduced 5 bit errors")
    print(f"   Packet valid after errors: {is_valid_after} (expected: False)")
    print(f"   After FEC correction, packet should be valid again")

    # Wrap stream in IP/UDP and save to PCAP
    print("\n6. Writing packet stream to PCAP:")
    ipudp_payloads = [build_ip_udp_wrapper(pkt_bytes) for pkt_bytes, _ in stream]
    pcap_file = "acm_packet_stream.pcap"
    frames_to_pcap(ipudp_payloads, filename=pcap_file)

    print("\n=== Ready for FEC integration ===")
