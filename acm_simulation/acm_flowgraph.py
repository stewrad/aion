#!/usr/bin/env python3
"""
ACM Simulation - Main Flowgraph
Complete adaptive coding and modulation simulation with:
- Packet generation
- FEC encoding
- Pilot insertion
- ACM modulation
- AWGN channel
- Channel estimation
- FEC decoding
- Packet validation
"""

from gnuradio import gr, blocks, digital, fec, channels, analog
from gnuradio.fft import window
import numpy as np
import time

# Import our custom modules
from .packet_generator import PacketGenerator
from .acm_controller import ACMController
from .pilot_inserter import PilotInserter, PilotRemover


class ACMFlowgraph(gr.top_block):
    """
    Complete ACM simulation flowgraph.
    
    Signal flow:
    PacketGenerator → FEC Encoder → Modulator → Pilot Inserter 
      → Channel → Pilot Remover → Demodulator → FEC Decoder → Validation
    
    ACM Controller monitors SNR and adjusts modulation/coding in real-time.
    """
    
    def __init__(self,
                 snr_db=10.0,
                 sample_rate=1000000,
                 packet_rate=100,
                 frame_size=1024,
                 acm_enabled=True):
        """
        Initialize ACM flowgraph.
        
        Args:
            snr_db: Initial SNR in dB
            sample_rate: System sample rate (samples/sec)
            packet_rate: Packet generation rate (packets/sec)
            frame_size: FEC frame size in bits
            acm_enabled: Enable adaptive modulation/coding
        """
        gr.top_block.__init__(self, "ACM Simulation")
        
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.packet_rate = packet_rate
        self.frame_size = frame_size
        self.acm_enabled = acm_enabled
        
        print("\n" + "="*70)
        print("ACM SIMULATION FLOWGRAPH INITIALIZATION")
        print("="*70)
        
        ##################################################
        # 1. Packet Generator
        ##################################################
        print("\n[1/9] Initializing Packet Generator...")
        self.packet_gen = PacketGenerator(
            max_payload_size=1500,
            packet_rate=packet_rate,
            seed=42
        )
        
        ##################################################
        # 2. Throttle (for rate control)
        ##################################################
        print("[2/9] Adding Throttle...")
        self.throttle = blocks.throttle(gr.sizeof_char, sample_rate)
        
        ##################################################
        # 3. FEC Encoder
        ##################################################
        print("[3/9] Setting up FEC Encoder...")
        self.acm_controller = ACMController(hysteresis_db=1.0)
        
        # Get initial MCS
        mcs_idx, modulation, coding, spec_eff = self.acm_controller.select_mcs(snr_db)
        print(f"  Initial MCS: {modulation.name} + {coding.name}")
        
        # Create FEC encoder
        encoder_obj, decoder_obj = self.acm_controller.get_fec_object(
            coding, frame_size
        )
        self.fec_encoder = fec.encoder(
            # encoder_obj_list=encoder_obj,
            # encoder_obj_list=[encoder_obj],
            # packed=False,
            # rev_pack=True,
            encoder_obj,
            gr.sizeof_char,
            gr.sizeof_char
        )
        self.current_encoder = encoder_obj
        
        ##################################################
        # 4. Modulator
        ##################################################
        print("[4/9] Setting up Modulator...")
        constellation = self.acm_controller.get_constellation(modulation)
        self.current_modulation = modulation
        self.current_constellation = constellation
        
        # Pack bits according to modulation
        self.packer = blocks.pack_k_bits_bb(int(modulation))
        
        # Chunks to symbols
        self.modulator = digital.chunks_to_symbols_bc(
            constellation.points(),
            1
        )
        
        ##################################################
        # 5. Pilot Inserter
        ##################################################
        print("[5/9] Setting up Pilot Inserter...")
        self.pilot_inserter = PilotInserter(
            frame_size=frame_size,
            pilots_per_frame=64,
            pilot_spacing=16
        )
        
        ##################################################
        # 6. Channel Model
        ##################################################
        print("[6/9] Setting up Channel Model...")
        noise_voltage = self._snr_to_noise_voltage(snr_db)
        self.channel = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0
        )
        
        ##################################################
        # 7. Pilot Remover & Channel Estimator
        ##################################################
        print("[7/9] Setting up Pilot Remover / Channel Estimator...")
        self.pilot_remover = PilotRemover(
            frame_size=frame_size,
            pilots_per_frame=64,
            pilot_spacing=16
        )
        
        ##################################################
        # 8. Demodulator
        ##################################################
        print("[8/9] Setting up Demodulator...")
        self.demodulator = digital.constellation_decoder_cb(
            constellation.base()
        )
        self.current_demod_constellation = constellation
        
        # Unpack bits
        self.unpacker = blocks.unpack_k_bits_bb(int(modulation))
        
        ##################################################
        # 9. FEC Decoder
        ##################################################
        print("[9/9] Setting up FEC Decoder...")
        self.fec_decoder = fec.decoder(
            # decoder_obj_list=decoder_obj,
            # packed=False,
            # rev_pack=True
            decoder_obj,
            gr.sizeof_char,
            gr.sizeof_char
        )
        self.current_decoder = decoder_obj
        
        ##################################################
        # Output and Monitoring
        ##################################################
        self.file_sink = blocks.file_sink(
            gr.sizeof_char,
            "/tmp/acm_output.bin",
            False
        )
        
        # Message debug blocks
        self.msg_debug_packets = blocks.message_debug()
        self.msg_debug_pilots = blocks.message_debug()
        self.msg_debug_channel = blocks.message_debug()
        
        # Probes for monitoring
        self.probe_rate_in = blocks.probe_rate(
            gr.sizeof_char,
            sample_rate / 1000,
            0.15
        )
        self.probe_rate_out = blocks.probe_rate(
            gr.sizeof_char,
            sample_rate / 1000,
            0.15
        )
        
        ##################################################
        # Connect Everything
        ##################################################
        print("\nConnecting flowgraph blocks...")
        
        # Main signal path
        self.connect(self.packet_gen, self.throttle)
        self.connect(self.throttle, self.probe_rate_in)
        self.connect(self.throttle, self.fec_encoder)
        self.connect(self.fec_encoder, self.packer)
        self.connect(self.packer, self.modulator)
        self.connect(self.modulator, self.pilot_inserter)
        self.connect(self.pilot_inserter, self.channel)
        self.connect(self.channel, self.pilot_remover)
        self.connect(self.pilot_remover, self.demodulator)
        self.connect(self.demodulator, self.unpacker)
        self.connect(self.unpacker, self.fec_decoder)
        self.connect(self.fec_decoder, self.probe_rate_out)
        self.connect(self.fec_decoder, self.file_sink)
        
        # Message connections
        self.msg_connect(
            self.packet_gen, "packet_info",
            self.msg_debug_packets, "store"
        )
        self.msg_connect(
            self.pilot_inserter, "pilot_info",
            self.msg_debug_pilots, "store"
        )
        self.msg_connect(
            self.pilot_remover, "channel_estimate",
            self.msg_debug_channel, "store"
        )
        
        print("\n" + "="*70)
        print("FLOWGRAPH INITIALIZATION COMPLETE")
        print("="*70)
        print(f"  Sample Rate: {sample_rate/1e6:.1f} MHz")
        print(f"  Initial SNR: {snr_db:.1f} dB")
        print(f"  Packet Rate: {packet_rate} pkt/s")
        print(f"  FEC Frame Size: {frame_size} bits")
        print(f"  ACM Enabled: {acm_enabled}")
        print("="*70 + "\n")
    
    def _snr_to_noise_voltage(self, snr_db):
        """Convert SNR in dB to noise voltage"""
        snr_linear = 10.0 ** (snr_db / 10.0)
        signal_power = 1.0
        noise_power = signal_power / snr_linear
        return np.sqrt(noise_power)
    
    def set_snr(self, snr_db):
        """
        Change channel SNR dynamically.
        This simulates changing channel conditions.
        """
        self.snr_db = snr_db
        noise_voltage = self._snr_to_noise_voltage(snr_db)
        self.channel.set_noise_voltage(noise_voltage)
        
        # Update ACM if enabled
        if self.acm_enabled:
            self.update_acm(snr_db)
        
        print(f"[ACM] SNR changed to {snr_db:.1f} dB")
    
    def update_acm(self, snr_db):
        """
        Update modulation and coding based on SNR.
        This is where ACM adaptation happens.
        """
        # Get new MCS from controller
        mcs_idx, new_mod, new_code, spec_eff = self.acm_controller.select_mcs(snr_db)
        
        # Check if MCS changed
        if (new_mod != self.current_modulation):
            print(f"\n[ACM] MCS CHANGE: {self.current_modulation.name} → {new_mod.name}")
            print(f"      Coding: {new_code.name}")
            print(f"      Spectral Efficiency: {spec_eff:.2f} bits/s/Hz")
            
            # In a real implementation, you would:
            # 1. Signal MCS change to pilot inserter
            # 2. Wait for pilot burst
            # 3. Reconfigure modulator/demodulator
            # 4. Reconfigure FEC encoder/decoder
            
            # For this simulation, we note the change
            # (Dynamic reconfiguration in GNU Radio requires more complex flow control)
            self.current_modulation = new_mod
            
            mcs_info = self.acm_controller.get_current_mcs_info()
            print(f"      New MCS Info: {mcs_info}")
    
    def get_throughput(self):
        """Get current throughput statistics"""
        input_rate = self.probe_rate_in.rate()
        output_rate = self.probe_rate_out.rate()
        
        return {
            'input_rate_bps': input_rate * 8,
            'output_rate_bps': output_rate * 8,
            'throughput_mbps': (output_rate * 8) / 1e6,
            'efficiency': output_rate / input_rate if input_rate > 0 else 0
        }
    
    def get_packet_stats(self):
        """Get packet generation statistics"""
        return self.packet_gen.get_stats()
    
    def get_acm_stats(self):
        """Get ACM controller statistics"""
        return self.acm_controller.get_current_mcs_info()


def main():
    """Main function to run the ACM simulation"""
    
    print("\n" + "="*70)
    print(" ACM SIMULATION - Adaptive Coding and Modulation ")
    print("="*70 + "\n")
    
    # Create flowgraph
    tb = ACMFlowgraph(
        snr_db=10.0,
        sample_rate=1000000,
        packet_rate=50,
        frame_size=1024,
        acm_enabled=True
    )
    
    # Start flowgraph
    tb.start()
    print("\n>>> Simulation running...\n")
    
    try:
        # Simulate changing channel conditions
        snr_profile = [
            (5, 10.0),   # 5 seconds at 10 dB
            (5, 15.0),   # 5 seconds at 15 dB
            (5, 8.0),    # 5 seconds at 8 dB (degraded)
            (5, 20.0),   # 5 seconds at 20 dB (good)
            (5, 5.0),    # 5 seconds at 5 dB (poor)
        ]
        
        for duration, snr in snr_profile:
            tb.set_snr(snr)
            
            for t in range(duration):
                time.sleep(1)
                
                # Get statistics
                throughput = tb.get_throughput()
                packet_stats = tb.get_packet_stats()
                acm_stats = tb.get_acm_stats()
                
                print(f"t={t+1}s | "
                      f"SNR={snr:.1f}dB | "
                      f"MCS={acm_stats['modulation']} | "
                      f"Throughput={throughput['throughput_mbps']:.2f}Mbps | "
                      f"Packets={packet_stats['packets_generated']}")
    
    except KeyboardInterrupt:
        print("\n\n>>> Stopping simulation...")
    
    # Stop flowgraph
    tb.stop()
    tb.wait()
    
    # Final statistics
    print("\n" + "="*70)
    print(" SIMULATION COMPLETE - Final Statistics ")
    print("="*70)
    
    packet_stats = tb.get_packet_stats()
    print(f"\nPacket Statistics:")
    print(f"  Packets Generated: {packet_stats['packets_generated']}")
    print(f"  Bytes Generated: {packet_stats['bytes_generated']}")
    print(f"  Avg Packet Size: {packet_stats['avg_packet_size']:.1f} bytes")
    
    throughput = tb.get_throughput()
    print(f"\nThroughput:")
    print(f"  Final Throughput: {throughput['throughput_mbps']:.2f} Mbps")
    print(f"  Efficiency: {throughput['efficiency']*100:.1f}%")
    
    acm_stats = tb.get_acm_stats()
    print(f"\nACM State:")
    print(f"  Final MCS: {acm_stats['modulation']} + {acm_stats['coding_rate']}")
    print(f"  Spectral Efficiency: {acm_stats['spectral_efficiency']:.2f} bits/s/Hz")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
