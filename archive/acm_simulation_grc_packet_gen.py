import numpy as np
from gnuradio import gr
import pmt
import struct
from enum import IntEnum

class PacketType(IntEnum):
    DATA = 0
    VOICE = 1
    VIDEO = 2
    CONTROL = 3

class blk(gr.sync_block):
    def __init__(self, packet_rate=100):
        gr.sync_block.__init__(self, name="packet_gen", in_sig=None, out_sig=[np.uint8])
        self.packet_rate = packet_rate
        self.seq_num = 0
        self.packet_buffer = np.array([], dtype=np.uint8)
        self.last_gen_time = 0
        self.message_port_register_out(pmt.intern("packet_info"))
    
    def _generate_packet(self, timestamp):
        header = struct.pack('!BBIdd', 1, 0, self.seq_num, timestamp, 100.0)
        payload = np.random.bytes(100)
        packet = header + payload + struct.pack('!I', 0xDEADBEEF)
        self.seq_num += 1
        
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern("seq_num"), pmt.from_long(self.seq_num-1))
        self.message_port_pub(pmt.intern("packet_info"), meta)
        
        return packet
    
    def work(self, input_items, output_items):
        out = output_items[0]
        current_time = self.nitems_written(0) / 1e6
        
        if self.last_gen_time == 0:
            self.last_gen_time = current_time
        
        time_elapsed = current_time - self.last_gen_time
        packets_to_generate = int(time_elapsed * self.packet_rate)
        
        if packets_to_generate > 0 and len(self.packet_buffer) < len(out):
            for _ in range(min(packets_to_generate, 5)):
                packet_bytes = self._generate_packet(current_time)
                packet_array = np.frombuffer(packet_bytes, dtype=np.uint8)
                packet_bits = np.unpackbits(packet_array)
                self.packet_buffer = np.append(self.packet_buffer, packet_bits)
            self.last_gen_time = current_time
        
        if len(self.packet_buffer) >= len(out):
            out[:] = self.packet_buffer[:len(out)]
            self.packet_buffer = self.packet_buffer[len(out):]
            return len(out)
        elif len(self.packet_buffer) > 0:
            n = len(self.packet_buffer)
            out[:n] = self.packet_buffer
            self.packet_buffer = np.array([], dtype=np.uint8)
            return n
        return 0
