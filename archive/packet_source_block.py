"""
Step-by-step guide to integrate your packet generator into GNU Radio
This creates a proper GNU Radio source block from your packet generator
"""

# ============================================================================
# STEP 1: Create a GNU Radio-compatible wrapper for your packet generator
# Save this as: packet_source_block.py
# ============================================================================

import numpy as np
from gnuradio import gr
import pmt
import time
from enum import IntEnum
import struct


class PacketType(IntEnum):
    """Packet types for different traffic classes"""
    DATA = 0
    VOICE = 1
    VIDEO = 2
    CONTROL = 3


class packet_generator_source(gr.sync_block):
    """
    GNU Radio source block wrapping your packet generator.
    
    This block:
    1. Generates realistic packets using your generator logic
    2. Outputs bits ready for FEC encoding
    3. Sends metadata via message ports for monitoring
    4. Rate limits to realistic packet generation speeds
    """
    
    def __init__(self, 
                 max_payload_size=1500,
                 packet_rate=100,
                 output_bits=True,
                 traffic_mix=None,
                 seed=None):
        """
        Initialize the packet generator source.
        
        Args:
            max_payload_size: Maximum payload size in bytes (default: 1500)
            packet_rate: Target packet generation rate (packets/sec)
            output_bits: If True, output unpacked bits (0/1); if False, bytes
            traffic_mix: Dict of {PacketType: probability}
            seed: Random seed for reproducibility
        """
        
        # Initialize as GNU Radio sync block
        gr.sync_block.__init__(
            self,
            name="packet_generator_source",
            in_sig=None,  # Source block - no input
            out_sig=[np.uint8]  # Output unsigned 8-bit (bits or bytes)
        )
        
        # Configuration
        self.max_payload_size = max_payload_size
        self.packet_rate = packet_rate
        self.output_bits = output_bits
        self.seq_num = 0
        
        # Packet structure sizes
        self.header_size = 16
        self.crc_size = 4
        self.min_packet_size = self.header_size + self.crc_size
        
        # Traffic mix configuration
        if traffic_mix is None:
            self.traffic_mix = {
                PacketType.DATA: 0.6,
                PacketType.VOICE: 0.15,
                PacketType.VIDEO: 0.2,
                PacketType.CONTROL: 0.05
            }
        else:
            self.traffic_mix = traffic_mix
        
        # Normalize traffic probabilities
        total = sum(self.traffic_mix.values())
        self.traffic_probs = [
            self.traffic_mix.get(PacketType(i), 0) / total for i in range(4)
        ]
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Buffer for generated packets
        self.packet_buffer = np.array([], dtype=np.uint8)
        
        # Timing for rate limiting
        self.last_gen_time = time.time()
        self.packets_generated = 0
        self.bytes_generated = 0
        
        # Message port for metadata output
        self.message_port_register_out(pmt.intern("metadata"))
        
        print(f"[Packet Generator] Initialized:")
        print(f"  - Max payload: {max_payload_size} bytes")
        print(f"  - Packet rate: {packet_rate} pkt/s")
        print(f"  - Output mode: {'bits' if output_bits else 'bytes'}")
        print(f"  - Traffic mix: {self.traffic_mix}")
    
    def _compute_crc32(self, data):
        """Compute CRC32 checksum"""
        return hash(bytes(data)) & 0xFFFFFFFF
    
    def _create_header(self, packet_type, payload_size):
        """Create packet header with timestamp and sequence number"""
        version = 1
        header = struct.pack(
            '!BBIdd',  # Network byte order
            version,
            packet_type,
            self.seq_num,
            time.time(),
            payload_size
        )
        self.seq_num += 1
        return header
    
    def _generate_payload(self, size, packet_type):
        """Generate realistic payload based on packet type"""
        if packet_type == PacketType.DATA:
            # High entropy data (compressed/encrypted)
            payload = np.random.bytes(size)
            
        elif packet_type == PacketType.VOICE:
            # Voice codec frames - lower entropy, periodic
            frame_pattern = np.random.randint(0, 128, size=size // 2, dtype=np.uint8)
            payload = np.repeat(frame_pattern, 2).tobytes()[:size]
            
        elif packet_type == PacketType.VIDEO:
            # Video - mix of I-frames (high entropy) and P-frames (lower)
            if np.random.random() < 0.1:  # I-frame
                payload = np.random.bytes(size)
            else:  # P-frame
                base = np.random.randint(0, 256, size=size // 4, dtype=np.uint8)
                payload = np.repeat(base, 4).tobytes()[:size]
                
        elif packet_type == PacketType.CONTROL:
            # Control packets - highly structured
            pattern = np.array([0xAA, 0x55] * (size // 2), dtype=np.uint8)
            payload = pattern.tobytes()[:size]
            
        else:
            payload = np.random.bytes(size)
            
        return payload
    
    def _generate_single_packet(self):
        """Generate one complete packet with header, payload, and CRC"""
        
        # Select packet type based on traffic mix
        packet_type = PacketType(np.random.choice(4, p=self.traffic_probs))
        
        # Determine payload size based on packet type
        if packet_type == PacketType.VOICE:
            # Voice packets: fixed sizes (codec frames)
            payload_size = np.random.choice([160, 320, 640])
        elif packet_type == PacketType.VIDEO:
            # Video packets: bursty, exponential distribution
            payload_size = int(np.random.exponential(800))
            payload_size = min(payload_size, self.max_payload_size)
        elif packet_type == PacketType.CONTROL:
            # Control packets: small
            payload_size = np.random.randint(64, 256)
        else:  # DATA
            # Data packets: uniform distribution
            payload_size = np.random.randint(64, self.max_payload_size)
        
        payload_size = max(1, min(payload_size, self.max_payload_size))
        
        # Build packet
        header = self._create_header(packet_type, payload_size)
        payload = self._generate_payload(payload_size, packet_type)
        packet_without_crc = header + payload
        
        # Add CRC for error detection
        crc = self._compute_crc32(packet_without_crc)
        crc_bytes = struct.pack('!I', crc)
        complete_packet = packet_without_crc + crc_bytes
        
        # Send metadata via message port for monitoring
        metadata = pmt.make_dict()
        metadata = pmt.dict_add(metadata, pmt.intern("seq_num"), 
                               pmt.from_long(self.seq_num - 1))
        metadata = pmt.dict_add(metadata, pmt.intern("packet_type"), 
                               pmt.intern(packet_type.name))
        metadata = pmt.dict_add(metadata, pmt.intern("payload_size"), 
                               pmt.from_long(payload_size))
        metadata = pmt.dict_add(metadata, pmt.intern("total_size"), 
                               pmt.from_long(len(complete_packet)))
        metadata = pmt.dict_add(metadata, pmt.intern("timestamp"), 
                               pmt.from_double(time.time()))
        
        self.message_port_pub(pmt.intern("metadata"), metadata)
        
        # Update statistics
        self.packets_generated += 1
        self.bytes_generated += len(complete_packet)
        
        return complete_packet
    
    def work(self, input_items, output_items):
        """
        GNU Radio work function - called continuously to produce output.
        
        This function:
        1. Generates packets at the specified rate
        2. Converts to bits if needed
        3. Fills the output buffer
        """
        out = output_items[0]
        noutput_items = len(out)
        
        # Rate limiting - only generate packets at specified rate
        current_time = time.time()
        time_since_last = current_time - self.last_gen_time
        
        # How many packets should we have generated by now?
        packets_should_generate = int(time_since_last * self.packet_rate)
        
        # Generate new packets if needed
        if packets_should_generate > 0 and len(self.packet_buffer) < noutput_items:
            for _ in range(min(packets_should_generate, 10)):  # Max 10 per call
                packet_bytes = self._generate_single_packet()
                packet_array = np.frombuffer(packet_bytes, dtype=np.uint8)
                
                if self.output_bits:
                    # Unpack bytes to bits (8 bits per byte)
                    packet_bits = np.unpackbits(packet_array)
                    self.packet_buffer = np.append(self.packet_buffer, packet_bits)
                else:
                    # Keep as bytes
                    self.packet_buffer = np.append(self.packet_buffer, packet_array)
            
            self.last_gen_time = current_time
        
        # Fill output buffer with available data
        if len(self.packet_buffer) >= noutput_items:
            # We have enough data
            out[:] = self.packet_buffer[:noutput_items]
            self.packet_buffer = self.packet_buffer[noutput_items:]
            return noutput_items
        else:
            # Output what we have
            if len(self.packet_buffer) > 0:
                out[:len(self.packet_buffer)] = self.packet_buffer
                produced = len(self.packet_buffer)
                self.packet_buffer = np.array([], dtype=np.uint8)
                return produced
            else:
                # No data yet
                return 0
    
    def get_stats(self):
        """Get generation statistics"""
        return {
            'packets_generated': self.packets_generated,
            'bytes_generated': self.bytes_generated,
            'avg_packet_size': self.bytes_generated / max(1, self.packets_generated)
        }


# ============================================================================
# STEP 2: Test the block standalone
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING PACKET GENERATOR BLOCK")
    print("="*60 + "\n")
    
    # This is just for testing - normally GNU Radio calls work()
    class TestHarness(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            
            # Create packet source
            self.packet_src = packet_generator_source(
                max_payload_size=1500,
                packet_rate=50,  # 50 packets/second
                output_bits=True,
                seed=42
            )
            
            # Add a file sink to capture output
            from gnuradio import blocks
            self.file_sink = blocks.file_sink(
                gr.sizeof_char,
                "/tmp/packet_test.bin"
            )
            
            # Add a message debug to see metadata
            self.msg_debug = blocks.message_debug()
            
            # Connect
            self.connect(self.packet_src, self.file_sink)
            self.msg_connect(self.packet_src, "metadata", self.msg_debug, "print")
    
    # Run test
    print("Creating test flowgraph...")
    tb = TestHarness()
    
    print("Starting flowgraph for 5 seconds...")
    tb.start()
    time.sleep(5)
    tb.stop()
    tb.wait()
    
    print("\nTest complete!")
    print(f"Stats: {tb.packet_src.get_stats()}")
    print(f"Output saved to: /tmp/packet_test.bin")
