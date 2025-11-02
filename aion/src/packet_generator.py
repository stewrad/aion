import numpy as np
import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple
import time


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


class PacketGenerator:
    """
    Realistic packet generator for ACM communication link simulation.
    Generates packets with proper headers, CRC, and realistic payload patterns.
    """
    
    def __init__(self, max_payload_size: int = 1500, seed: int = None):
        """
        Initialize packet generator.
        
        Args:
            max_payload_size: Maximum payload size in bytes (default: 1500 for Ethernet MTU)
            seed: Random seed for reproducibility
        """
        self.max_payload_size = max_payload_size
        self.seq_num = 0
        if seed is not None:
            np.random.seed(seed)
        
        # Header structure: version(1) + type(1) + seq(4) + timestamp(8) + length(2) = 16 bytes
        self.header_size = 16
        self.crc_size = 4
        self.min_packet_size = self.header_size + self.crc_size
        
    def _compute_crc32(self, data: bytes) -> int:
        """Compute CRC32 checksum for error detection"""
        return np.uint32(np.bitwise_xor.reduce(
            np.frombuffer(data, dtype=np.uint8).view(np.uint32)
        )) if len(data) % 4 == 0 else hash(data) & 0xFFFFFFFF
    
    def _create_header(self, packet_type: PacketType, payload_size: int) -> bytes:
        """
        Create packet header.
        
        Format:
        - Version (1 byte): Protocol version
        - Type (1 byte): Packet type
        - Sequence (4 bytes): Packet sequence number
        - Timestamp (8 bytes): Double precision timestamp
        - Length (2 bytes): Payload length
        """
        version = 1
        header = struct.pack(
            '!BBIdd',  # Network byte order (big-endian)
            version,
            packet_type,
            self.seq_num,
            time.time(),
            payload_size
        )
        self.seq_num += 1
        return header
    
    def _generate_realistic_payload(self, size: int, packet_type: PacketType) -> bytes:
        """
        Generate realistic payload data based on packet type.
        
        Args:
            size: Payload size in bytes
            packet_type: Type of packet to generate
            
        Returns:
            Payload bytes with realistic characteristics
        """
        if packet_type == PacketType.DATA:
            # Mix of text-like data (high entropy) and structured data
            # Simulate compressed data with moderate entropy
            payload = np.random.bytes(size)
            
        elif packet_type == PacketType.VOICE:
            # Voice packets: periodic structure, lower entropy
            # Simulate codec frames (e.g., G.711, Opus)
            frame_pattern = np.random.randint(0, 128, size=size // 2, dtype=np.uint8)
            payload = np.repeat(frame_pattern, 2).tobytes()[:size]
            
        elif packet_type == PacketType.VIDEO:
            # Video packets: bursty, high variance in size
            # Mix of high entropy (I-frames) and low entropy (P-frames)
            if np.random.random() < 0.1:  # I-frame (10% of packets)
                payload = np.random.bytes(size)
            else:  # P-frame with more structure
                base = np.random.randint(0, 256, size=size // 4, dtype=np.uint8)
                payload = np.repeat(base, 4).tobytes()[:size]
                
        elif packet_type == PacketType.CONTROL:
            # Control packets: highly structured, low entropy
            # Simulate protocol messages with fixed patterns
            pattern = np.array([0xAA, 0x55] * (size // 2), dtype=np.uint8)
            payload = pattern.tobytes()[:size]
        
        else:
            payload = np.random.bytes(size)
            
        return payload
    
    def generate_packet(self, payload_size: int = None, 
                       packet_type: PacketType = PacketType.DATA) -> Tuple[bytes, PacketMetadata]:
        """
        Generate a single packet with header, payload, and CRC.
        
        Args:
            payload_size: Size of payload in bytes (random if None)
            packet_type: Type of packet to generate
            
        Returns:
            Tuple of (complete_packet_bytes, metadata)
        """
        if payload_size is None:
            # Generate random size based on packet type
            if packet_type == PacketType.VOICE:
                payload_size = np.random.choice([160, 320, 640])  # Common voice frame sizes
            elif packet_type == PacketType.VIDEO:
                payload_size = int(np.random.exponential(800))  # Bursty video
                payload_size = min(payload_size, self.max_payload_size)
            elif packet_type == PacketType.CONTROL:
                payload_size = np.random.randint(64, 256)
            else:  # DATA
                payload_size = np.random.randint(64, self.max_payload_size)
        
        payload_size = min(payload_size, self.max_payload_size)
        
        # Create packet components
        header = self._create_header(packet_type, payload_size)
        payload = self._generate_realistic_payload(payload_size, packet_type)
        
        # Compute CRC over header + payload
        packet_without_crc = header + payload
        crc = self._compute_crc32(packet_without_crc)
        crc_bytes = struct.pack('!I', crc)
        
        # Complete packet
        complete_packet = packet_without_crc + crc_bytes
        
        # Create metadata
        metadata = PacketMetadata(
            seq_num=self.seq_num - 1,
            timestamp=time.time(),
            packet_type=packet_type,
            payload_size=payload_size,
            priority=int(packet_type)
        )
        
        return complete_packet, metadata
    
    def generate_packet_stream(self, num_packets: int, 
                              traffic_mix: dict = None) -> List[Tuple[bytes, PacketMetadata]]:
        """
        Generate a stream of packets with realistic traffic mix.
        
        Args:
            num_packets: Number of packets to generate
            traffic_mix: Dict mapping PacketType to probability (default: realistic mix)
            
        Returns:
            List of (packet, metadata) tuples
        """
        if traffic_mix is None:
            # Default realistic traffic mix for satcom
            traffic_mix = {
                PacketType.DATA: 0.6,
                PacketType.VOICE: 0.15,
                PacketType.VIDEO: 0.2,
                PacketType.CONTROL: 0.05
            }
        
        # Normalize probabilities
        total = sum(traffic_mix.values())
        traffic_probs = [traffic_mix.get(PacketType(i), 0) / total for i in range(4)]
        
        packets = []
        for _ in range(num_packets):
            packet_type = PacketType(np.random.choice(4, p=traffic_probs))
            packet, metadata = self.generate_packet(packet_type=packet_type)
            packets.append((packet, metadata))
        
        return packets
    
    def verify_packet(self, packet: bytes) -> Tuple[bool, dict]:
        """
        Verify packet integrity (for testing FEC decoding).
        
        Args:
            packet: Complete packet bytes
            
        Returns:
            Tuple of (is_valid, parsed_info)
        """
        if len(packet) < self.min_packet_size:
            return False, {"error": "Packet too short"}
        
        try:
            # Extract components
            header = packet[:self.header_size]
            crc_received = struct.unpack('!I', packet[-self.crc_size:])[0]
            packet_without_crc = packet[:-self.crc_size]
            
            # Verify CRC
            crc_computed = self._compute_crc32(packet_without_crc)
            is_valid = (crc_received == crc_computed)
            
            # Parse header
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
        """
        Convert packet to bit array for FEC encoding.
        
        Args:
            packet: Packet bytes
            
        Returns:
            NumPy array of bits (0s and 1s)
        """
        return np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    
    def from_bit_array(self, bits: np.ndarray) -> bytes:
        """
        Convert bit array back to packet bytes (after FEC decoding).
        
        Args:
            bits: NumPy array of bits
            
        Returns:
            Packet bytes
        """
        # Pad to multiple of 8 if necessary
        remainder = len(bits) % 8
        if remainder != 0:
            bits = np.append(bits, np.zeros(8 - remainder, dtype=np.uint8))
        
        return np.packbits(bits).tobytes()


# Example usage and testing
if __name__ == "__main__":
    print("=== ACM Packet Generator Demo ===\n")
    
    # Initialize generator
    gen = PacketGenerator(max_payload_size=1500, seed=42)
    
    # Generate single packets of different types
    print("1. Generating individual packets:")
    for pkt_type in PacketType:
        packet, metadata = gen.generate_packet(packet_type=pkt_type)
        print(f"   {pkt_type.name}: {len(packet)} bytes, "
              f"seq={metadata.seq_num}, payload={metadata.payload_size} bytes")
    
    # Generate packet stream
    print("\n2. Generating packet stream (100 packets):")
    stream = gen.generate_packet_stream(100)
    
    # Statistics
    type_counts = {}
    total_bytes = 0
    for packet, metadata in stream:
        type_counts[metadata.packet_type.name] = type_counts.get(metadata.packet_type.name, 0) + 1
        total_bytes += len(packet)
    
    print(f"   Total bytes: {total_bytes}")
    print(f"   Traffic distribution: {type_counts}")
    print(f"   Average packet size: {total_bytes / len(stream):.1f} bytes")
    
    # Verify packets
    print("\n3. Verifying packet integrity:")
    test_packet, _ = gen.generate_packet(payload_size=500, packet_type=PacketType.DATA)
    is_valid, info = gen.verify_packet(test_packet)
    print(f"   Packet valid: {is_valid}")
    print(f"   Parsed info: {info}")
    
    # Convert to bits for FEC
    print("\n4. Converting to bit array for FEC encoding:")
    bits = gen.to_bit_array(test_packet)
    print(f"   Packet size: {len(test_packet)} bytes = {len(bits)} bits")
    
    # Simulate bit errors and recovery
    print("\n5. Simulating bit errors (for FEC testing):")
    corrupted_bits = bits.copy()
    error_positions = np.random.choice(len(bits), size=5, replace=False)
    corrupted_bits[error_positions] = 1 - corrupted_bits[error_positions]
    
    recovered_packet = gen.from_bit_array(corrupted_bits)
    is_valid_after, _ = gen.verify_packet(recovered_packet)
    print(f"   Introduced 5 bit errors")
    print(f"   Packet valid after errors: {is_valid_after} (expected: False)")
    print(f"   After FEC correction, packet should be valid again")
    
    print("\n=== Ready for FEC integration ===")

    # ============================================================
    # 6. Export generated packet stream to PCAP for Wireshark
    # ============================================================
    print("\n6. Writing packet stream to PCAP file (for Wireshark):")

    from scapy.all import Ether, Raw, wrpcap, IP, UDP

    # Wrap each generated packet in a fake Ethernet frame
    scapy_packets = []
    for packet, meta in stream:
        # Ether + Raw payload; Ethernet is just a dummy header for Wireshark compatibility
        scapy_pkt = Ether() / IP(src="192.168.0.1", dst="192.168.0.2") / UDP(sport=5000, dport=6000) / Raw(packet)
        scapy_packets.append(scapy_pkt)

    # Save to file
    pcap_filename = "acm_generated_packets.pcap"
    wrpcap(pcap_filename, scapy_packets)
    print(f"   ✅ Saved {len(scapy_packets)} packets to '{pcap_filename}'")

    print("\nYou can now open this file in Wireshark to inspect packet contents.")
