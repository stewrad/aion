"""
Complete GNU Radio Integration Example for ACM Simulation
Step-by-step guide with working flowgraph
"""

from gnuradio import gr, blocks, digital, fec
from gnuradio import analog, channels
import numpy as np
import time

# Import our custom blocks (assuming packet_blocks.py is in the same directory)
# from packet_blocks import packet_source, packet_sink


class acm_simulation_flowgraph(gr.top_block):
    """
    Complete ACM simulation flowgraph demonstrating the packet generator integration.
    
    Signal Flow:
    Packet Source → FEC Encoder → BPSK Modulator → AWGN Channel 
      → BPSK Demodulator → FEC Decoder → Packet Sink
    
    This is a baseline for comparing ACM vs ACM+RL performance.
    """
    
    def __init__(self, 
                 snr_db=10.0,
                 sample_rate=1e6,
                 packet_rate=100):
        """
        Initialize the ACM simulation flowgraph.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            sample_rate: Sample rate in Hz
            packet_rate: Packet generation rate (packets/sec)
        """
        gr.top_block.__init__(self, "ACM Simulation")
        
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.packet_rate = packet_rate
        
        ##################################################
        # Step 1: Create Packet Source
        ##################################################
        print("Step 1: Creating packet source...")
        
        # Import the packet source (you'll need to uncomment the import above)
        # For this example, we'll describe what it does:
        # - Generates realistic packets with different traffic types
        # - Outputs unpacked bits (0s and 1s as uint8)
        # - Rate limited to packet_rate packets/sec
        
        # self.packet_src = packet_source(
        #     max_payload_size=1500,
        #     packets_per_burst=10,
        #     output_bits=True,  # Output bits for FEC
        #     packet_rate=packet_rate,
        #     seed=42
        # )
        
        # For demonstration, use a random source instead
        self.packet_src = analog.random_uniform_source_b(0, 2, seed=42)
        
        ##################################################
        # Step 2: FEC Encoding
        ##################################################
        print("Step 2: Setting up FEC encoder...")
        
        # Choose an FEC code - for ACM, you'd switch between these
        # Common options for satcom:
        # - LDPC codes (various rates: 1/2, 2/3, 3/4, 5/6)
        # - Turbo codes
        # - Convolutional codes with different constraint lengths
        
        # Example: LDPC encoder with rate 1/2
        fec_encoder_ldpc = fec.ldpc_encoder_make(
            fec.ldpc_H_matrix('/usr/share/gnuradio/fec/ldpc/n_1800_k_0902_gap_28.alist'),
            repetitions=1
        )
        
        # For simpler testing, use convolutional code
        fec_encoder_cc = fec.cc_encoder_make(
            frame_size=1024,  # Must match your packet size
            k=7,  # Constraint length
            rate=2,  # Code rate (1/rate)
            polys=[79, 109]  # Polynomials for rate 1/2
        )
        
        # Create the async encoder block
        self.fec_enc = fec.async_encoder(
            encoder_obj_list=fec_encoder_cc,
            packed=False,  # Input is unpacked bits
            rev_pack=True
        )
        
        ##################################################
        # Step 3: Modulation
        ##################################################
        print("Step 3: Setting up modulator...")
        
        # For ACM, you would dynamically switch between modulation schemes:
        # - BPSK (most robust, lowest rate)
        # - QPSK (2 bits/symbol)
        # - 8PSK (3 bits/symbol)
        # - 16QAM (4 bits/symbol)
        # - 64QAM (6 bits/symbol)
        
        # Start with BPSK for simplicity
        self.constellation = digital.constellation_bpsk()
        
        # Modulator - converts bits to complex symbols
        self.modulator = digital.chunks_to_symbols_bc(
            self.constellation.points(),
            1  # Dimension
        )
        
        ##################################################
        # Step 4: Channel Model
        ##################################################
        print("Step 4: Adding channel model...")
        
        # AWGN Channel - additive white Gaussian noise
        noise_voltage = self._calculate_noise_voltage(snr_db)
        self.channel = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,  # Timing offset
            taps=[1.0],  # No multipath
            noise_seed=0
        )
        
        # For more realistic satcom simulation, you could add:
        # - Doppler shift (frequency_offset parameter)
        # - Multipath fading (taps parameter)
        # - Phase noise
        
        ##################################################
        # Step 5: Demodulation
        ##################################################
        print("Step 5: Setting up demodulator...")
        
        # Demodulator - converts symbols back to bits
        self.demodulator = digital.constellation_decoder_cb(
            self.constellation.base()
        )
        
        ##################################################
        # Step 6: FEC Decoding
        ##################################################
        print("Step 6: Setting up FEC decoder...")
        
        # Corresponding decoder
        fec_decoder_cc = fec.cc_decoder.make(
            frame_size=1024,
            k=7,
            rate=2,
            polys=[79, 109],
            mode=fec.CC_STREAMING,
            padding=False
        )
        
        self.fec_dec = fec.async_decoder(
            decoder_obj_list=fec_decoder_cc,
            packed=False,
            rev_pack=True
        )
        
        ##################################################
        # Step 7: Packet Sink (Validation)
        ##################################################
        print("Step 7: Creating packet sink...")
        
        # self.packet_sink = packet_sink(
        #     input_bits=True,
        #     packet_size_estimate=1600
        # )
        
        # For demonstration, use a file sink
        self.file_sink = blocks.file_sink(
            gr.sizeof_char,
            "/tmp/received_packets.bin",
            False
        )
        
        ##################################################
        # Step 8: Connect Everything
        ##################################################
        print("Step 8: Connecting flowgraph...")
        
        # Source → FEC Encoder
        self.connect((self.packet_src, 0), (self.fec_enc, 0))
        
        # FEC Encoder → Modulator
        self.connect((self.fec_enc, 0), (self.modulator, 0))
        
        # Modulator → Channel
        self.connect((self.modulator, 0), (self.channel, 0))
        
        # Channel → Demodulator
        self.connect((self.channel, 0), (self.demodulator, 0))
        
        # Demodulator → FEC Decoder
        self.connect((self.demodulator, 0), (self.fec_dec, 0))
        
        # FEC Decoder → Packet Sink
        self.connect((self.fec_dec, 0), (self.file_sink, 0))
        
        ##################################################
        # Step 9: Add Instrumentation
        ##################################################
        print("Step 9: Adding measurement probes...")
        
        # Add probes for monitoring
        self.probe_rate_src = blocks.probe_rate(
            gr.sizeof_char,
            sample_rate / 1000,
            0.15
        )
        self.connect((self.packet_src, 0), (self.probe_rate_src, 0))
        
        self.probe_rate_sink = blocks.probe_rate(
            gr.sizeof_char,
            sample_rate / 1000,
            0.15
        )
        self.connect((self.fec_dec, 0), (self.probe_rate_sink, 0))
        
        print("Flowgraph created successfully!\n")
    
    def _calculate_noise_voltage(self, snr_db):
        """Calculate noise voltage for desired SNR"""
        snr_linear = 10.0 ** (snr_db / 10.0)
        signal_power = 1.0  # Normalized
        noise_power = signal_power / snr_linear
        noise_voltage = np.sqrt(noise_power)
        return noise_voltage
    
    def set_snr(self, snr_db):
        """Dynamically change SNR (for ACM adaptation)"""
        self.snr_db = snr_db
        noise_voltage = self._calculate_noise_voltage(snr_db)
        self.channel.set_noise_voltage(noise_voltage)
        print(f"SNR changed to {snr_db} dB")
    
    def get_throughput(self):
        """Get current throughput metrics"""
        src_rate = self.probe_rate_src.rate()
        sink_rate = self.probe_rate_sink.rate()
        
        return {
            'source_rate_bps': src_rate * 8,
            'sink_rate_bps': sink_rate * 8,
            'efficiency': sink_rate / src_rate if src_rate > 0 else 0
        }


##################################################
# Step-by-Step Usage Instructions
##################################################

def print_usage_instructions():
    """Print detailed usage instructions"""
    
    instructions = """
=================================================================
    GNU RADIO INTEGRATION - STEP BY STEP GUIDE
=================================================================

STEP 1: FILE ORGANIZATION
--------------------------
Create this directory structure:
    acm_project/
    ├── packet_generator.py       # Original packet generator from first artifact
    ├── packet_blocks.py          # GNU Radio blocks from second artifact  
    ├── acm_simulation.py         # This file
    └── analysis/
        └── results.txt

STEP 2: INSTALL DEPENDENCIES
------------------------------
sudo apt-get install gnuradio
pip install numpy

STEP 3: PREPARE PACKET BLOCKS
-------------------------------
1. Copy packet_blocks.py to your gnuradio module path:
   
   cp packet_blocks.py ~/.grc_gnuradio/
   
   OR add to PYTHONPATH:
   
   export PYTHONPATH=$PYTHONPATH:/path/to/acm_project

STEP 4: CREATE SIMPLE FLOWGRAPH (Python)
------------------------------------------
from acm_simulation import acm_simulation_flowgraph
import time

# Create flowgraph
tb = acm_simulation_flowgraph(
    snr_db=10.0,
    sample_rate=1e6,
    packet_rate=100
)

# Start flowgraph
tb.start()
print("Simulation running...")

# Run for 10 seconds
time.sleep(10)

# Get statistics
stats = tb.get_throughput()
print(f"Throughput: {stats['sink_rate_bps']/1e6:.2f} Mbps")
print(f"Efficiency: {stats['efficiency']*100:.1f}%")

# Stop
tb.stop()
tb.wait()

STEP 5: CREATE FLOWGRAPH IN GNU RADIO COMPANION
------------------------------------------------
1. Open GNU Radio Companion (GRC)

2. Add these blocks:
   - Import: "from packet_blocks import packet_source, packet_sink"
   - Python Block (packet_source):
     * Parameters:
       - max_payload_size: 1500
       - packets_per_burst: 10
       - output_bits: True
       - packet_rate: 100
   
   - FEC Extended Async Encoder:
     * Encoder Object: LDPC or CC Encoder
     * Frame Size: 1024
   
   - Chunks to Symbols:
     * Symbol Table: Choose constellation (BPSK/QPSK/QAM)
   
   - Channel Model:
     * Noise Voltage: 0.1 (adjust for desired SNR)
     * Frequency Offset: 0
   
   - Constellation Decoder:
     * Constellation Object: Match your modulator
   
   - FEC Extended Async Decoder:
     * Decoder Object: Match your encoder
   
   - Python Block (packet_sink):
     * Parameters:
       - input_bits: True

3. Connect blocks in order (as shown above)

4. Add Message Debug blocks connected to:
   - packet_source → metadata port
   - packet_sink → packets port

5. Generate and run!

STEP 6: IMPLEMENT ACM LOGIC
----------------------------
For adaptive coding and modulation, create a controller:

class ACMController:
    def __init__(self, flowgraph):
        self.fg = flowgraph
        self.snr_estimate = 10.0
        
    def update_link_params(self):
        # Measure channel quality
        stats = self.fg.get_throughput()
        
        # Simple ACM decision logic
        if stats['efficiency'] < 0.5:
            # Channel degraded, use more robust mode
            self.fg.set_snr(self.snr_estimate - 2)
        elif stats['efficiency'] > 0.9:
            # Channel good, use higher throughput mode
            self.fg.set_snr(self.snr_estimate + 2)

# Usage:
tb = acm_simulation_flowgraph()
controller = ACMController(tb)
tb.start()

while running:
    time.sleep(1)
    controller.update_link_params()

STEP 7: ADD RL AGENT (Future)
-------------------------------
Replace ACMController with your RL agent:

class RLACMController:
    def __init__(self, flowgraph):
        self.fg = flowgraph
        self.agent = YourRLAgent()
        
    def step(self):
        # Get state from flowgraph
        state = {
            'snr': self.fg.snr_db,
            'throughput': self.fg.get_throughput(),
            'packet_loss': ...
        }
        
        # Get action from RL agent
        action = self.agent.select_action(state)
        
        # Apply action (change modulation, coding rate, etc.)
        self.apply_action(action)
        
        # Calculate reward
        reward = self.calculate_reward(state)
        
        # Train agent
        self.agent.train(state, action, reward)

STEP 8: COLLECT METRICS FOR COMPARISON
----------------------------------------
import pandas as pd

metrics = {
    'timestamp': [],
    'snr_db': [],
    'modulation': [],
    'code_rate': [],
    'throughput_mbps': [],
    'packet_loss_rate': [],
    'latency_ms': []
}

# During simulation, log metrics
while running:
    stats = tb.get_throughput()
    metrics['timestamp'].append(time.time())
    metrics['snr_db'].append(tb.snr_db)
    metrics['throughput_mbps'].append(stats['sink_rate_bps']/1e6)
    # ... etc

# Save for analysis
df = pd.DataFrame(metrics)
df.to_csv('acm_vs_rl_results.csv')

STEP 9: VISUALIZE RESULTS
--------------------------
import matplotlib.pyplot as plt

# Compare ACM vs ACM+RL
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_acm['timestamp'], df_acm['throughput_mbps'], label='ACM')
plt.plot(df_rl['timestamp'], df_rl['throughput_mbps'], label='ACM+RL')
plt.xlabel('Time (s)')
plt.ylabel('Throughput (Mbps)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df_acm['snr_db'], df_acm['throughput_mbps'], 'o', label='ACM')
plt.plot(df_rl['snr_db'], df_rl['throughput_mbps'], 'x', label='ACM+RL')
plt.xlabel('SNR (dB)')
plt.ylabel('Throughput (Mbps)')
plt.legend()
plt.show()

=================================================================
TROUBLESHOOTING
=================================================================

Issue: "ImportError: No module named packet_blocks"
Solution: Check PYTHONPATH or copy to ~/.grc_gnuradio/

Issue: FEC encoder/decoder frame size mismatch
Solution: Ensure packet sizes are multiples of FEC frame size

Issue: Flowgraph not stopping cleanly
Solution: Always call tb.stop() then tb.wait()

Issue: Poor performance
Solution: Start with BPSK + rate 1/2 code, increase complexity gradually

=================================================================
NEXT STEPS FOR YOUR PROJECT
=================================================================

1. ✓ Packet generator created
2. ✓ GNU Radio blocks integrated
3. → Test with simple AWGN channel
4. → Add channel estimation
5. → Implement ACM decision logic
6. → Collect baseline ACM performance data
7. → Design RL state/action/reward
8. → Train RL agent
9. → Compare ACM vs ACM+RL performance

Good luck with your project!
"""
    
    print(instructions)


if __name__ == "__main__":
    print_usage_instructions()
    
    # Example: Run a simple simulation
    print("\n" + "="*60)
    print("RUNNING EXAMPLE SIMULATION")
    print("="*60 + "\n")
    
    try:
        tb = acm_simulation_flowgraph(
            snr_db=8.0,
            sample_rate=1e6,
            packet_rate=50
        )
        
        tb.start()
        print("Simulation running for 5 seconds...\n")
        
        for i in range(5):
            time.sleep(1)
            stats = tb.get_throughput()
            print(f"t={i+1}s: Throughput={stats['sink_rate_bps']/1e6:.2f} Mbps, "
                  f"Efficiency={stats['efficiency']*100:.1f}%")
        
        tb.stop()
        tb.wait()
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        print("This is expected if GNU Radio is not installed.")
        print("Follow the steps above to set up your environment.")