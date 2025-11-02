"""
ACM Simulation - Packet Generator Module
Generates realistic packet data for transmission
"""

import numpy as np
import struct
from enum import IntEnum
from gnuradio import gr
import pmt


class PacketType(IntEnum):
    """Packet types for different traffic classes"""
    DATA = 0
    VOICE = 1
    VIDEO = 2
    CONTROL = 3


class PacketGenerator(gr.sync_block):
    """
    GNU Radio source block that generates realistic packet data.
    Outputs unpacked bits ready for FEC encoding.
    """
    
    def __init__(self, 
                 max_payload_size=1500,
                 packet_rate=100,
                 traffic_mix=None,
                 seed=None):
        """
        Initialize packet generator.
        
        Args:
            max_payload_size: Maximum payload size in bytes
            packet_rate: Target packet generation rate (packets/sec)
            traffic_mix: Dict of {PacketType: probability}
            seed: Random seed for reproducibility
        """
        gr.sync_block.__init__(
            self,
            name="packet_generator",
            in_sig=None,
            out_sig=[np.uint8]  # Output bits
        )
        
        self.max_payload_size = max_payload_size
        self.packet_rate = packet_rate
        self.seq_num = 0
        
        # Packet structure
        self.header_size = 16  # bytes
        self.crc_size = 4      # bytes
        
        # Traffic mix
        if traffic_mix is None:
            self.traffic_mix = {
                PacketType.DATA: 0.60,
                PacketType.VOICE: 0.15,
                PacketType.VIDEO: 0.20,
                PacketType.CONTROL: 0.05
            }
        else:
            self.traffic_mix = traffic_mix
        
        # Normalize probabilities
        total = sum(self.traffic_mix.values())
        self.traffic_probs = [
            self.traffic_mix.get(PacketType(i), 0) / total 
            for i in range(4)
        ]
        
        if seed is not None:
            np.random.seed(seed)
        
        # Buffers
        self.packet_buffer = np.array([], dtype=np.uint8)
        self.last_gen_time = 0
        
        # Statistics
        self.packets_generated = 0
        self.bytes_generated = 0
        
        # Message port for metadata
        self.message_port_register_out(pmt.intern("packet_info"))
        
        print(f"[PacketGenerator] Initialized: rate={packet_rate} pkt/s, "
              f"max_size={max_payload_size} bytes")
    
    def _create_header(self, packet_type, payload_size, timestamp):
        """Create packet header"""
        version = 1
        header = struct.pack(
            '!BBIdd',  # Network byte order
            version,
            int(packet_type),
            self.seq_num,
            timestamp,
            float(payload_size)
        )
        self.seq_num += 1
        return header
    
    def _generate_payload(self, size, packet_type):
        """Generate realistic payload based on type"""
        size = max(1, size)
        
        if packet_type == PacketType.DATA:
            return np.random.bytes(size)
        elif packet_type == PacketType.VOICE:
            frame = np.random.randint(0, 128, size=size//2, dtype=np.uint8)
            return np.repeat(frame, 2).tobytes()[:size]
        elif packet_type == PacketType.VIDEO:
            if np.random.random() < 0.1:  # I-frame
                return np.random.bytes(size)
            else:  # P-frame
                base = np.random.randint(0, 256, size=size//4, dtype=np.uint8)
                return np.repeat(base, 4).tobytes()[:size]
        elif packet_type == PacketType.CONTROL:
            pattern = np.array([0xAA, 0x55] * (size//2), dtype=np.uint8)
            return pattern.tobytes()[:size]
        else:
            return np.random.bytes(size)
    
    def _compute_crc32(self, data):
        """Compute CRC32 checksum"""
        return hash(bytes(data)) & 0xFFFFFFFF
    
    def _generate_packet(self, timestamp):
        """Generate a complete packet"""
        # Select packet type
        packet_type = PacketType(np.random.choice(4, p=self.traffic_probs))
        
        # Determine payload size
        if packet_type == PacketType.VOICE:
            payload_size = np.random.choice([160, 320, 640])
        elif packet_type == PacketType.VIDEO:
            payload_size = int(np.random.exponential(800))
            payload_size = min(payload_size, self.max_payload_size)
        elif packet_type == PacketType.CONTROL:
            payload_size = np.random.randint(64, 256)
        else:  # DATA
            payload_size = np.random.randint(64, self.max_payload_size)
        
        payload_size = max(1, min(payload_size, self.max_payload_size))
        
        # Build packet
        header = self._create_header(packet_type, payload_size, timestamp)
        payload = self._generate_payload(payload_size, packet_type)
        packet_without_crc = header + payload
        
        # Add CRC
        crc = self._compute_crc32(packet_without_crc)
        crc_bytes = struct.pack('!I', crc)
        complete_packet = packet_without_crc + crc_bytes
        
        # Publish metadata
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern("seq_num"), 
                           pmt.from_long(self.seq_num - 1))
        meta = pmt.dict_add(meta, pmt.intern("packet_type"), 
                           pmt.intern(packet_type.name))
        meta = pmt.dict_add(meta, pmt.intern("payload_size"), 
                           pmt.from_long(payload_size))
        meta = pmt.dict_add(meta, pmt.intern("total_size"), 
                           pmt.from_long(len(complete_packet)))
        meta = pmt.dict_add(meta, pmt.intern("timestamp"), 
                           pmt.from_double(timestamp))
        self.message_port_pub(pmt.intern("packet_info"), meta)
        
        # Update statistics
        self.packets_generated += 1
        self.bytes_generated += len(complete_packet)
        
        return complete_packet
    
    def work(self, input_items, output_items):
        """GNU Radio work function"""
        out = output_items[0]
        noutput_items = len(out)
        
        # Rate limiting
        current_time = self.nitems_written(0) / 1e6  # Approximate time
        if self.last_gen_time == 0:
            self.last_gen_time = current_time
        
        time_elapsed = current_time - self.last_gen_time
        packets_to_generate = int(time_elapsed * self.packet_rate)
        
        # Generate packets
        if packets_to_generate > 0 and len(self.packet_buffer) < noutput_items:
            for _ in range(min(packets_to_generate, 10)):
                packet_bytes = self._generate_packet(current_time)
                packet_array = np.frombuffer(packet_bytes, dtype=np.uint8)
                # Unpack to bits
                packet_bits = np.unpackbits(packet_array)
                self.packet_buffer = np.append(self.packet_buffer, packet_bits)
            self.last_gen_time = current_time
        
        # Fill output
        if len(self.packet_buffer) >= noutput_items:
            out[:] = self.packet_buffer[:noutput_items]
            self.packet_buffer = self.packet_buffer[noutput_items:]
            return noutput_items
        elif len(self.packet_buffer) > 0:
            n = len(self.packet_buffer)
            out[:n] = self.packet_buffer
            self.packet_buffer = np.array([], dtype=np.uint8)
            return n
        else:
            return 0
    
    def get_stats(self):
        """Return generation statistics"""
        return {
            'packets_generated': self.packets_generated,
            'bytes_generated': self.bytes_generated,
            'avg_packet_size': (self.bytes_generated / max(1, self.packets_generated))
        }
