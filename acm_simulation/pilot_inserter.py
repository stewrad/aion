"""
ACM Simulation - Pilot Symbol Inserter
Inserts pilot symbols for channel estimation and frame synchronization
Save as: acm_simulation/pilot_inserter.py
"""

import numpy as np
from gnuradio import gr
import pmt


class PilotInserter(gr.sync_block):
    """
    Inserts pilot symbols at:
    1. Start of each frame (frame sync + channel estimation)
    2. Before each MCS change (to estimate channel for new MCS)
    
    Pilot pattern: Known BPSK sequence for robust channel estimation
    """
    
    def __init__(self,
                 frame_size=1024,
                 pilots_per_frame=64,
                 pilot_spacing=16):
        """
        Initialize pilot inserter.
        
        Args:
            frame_size: Size of data frame in symbols
            pilots_per_frame: Number of pilot symbols at frame start
            pilot_spacing: Insert one pilot every N data symbols
        """
        gr.sync_block.__init__(
            self,
            name="pilot_inserter",
            in_sig=[np.complex64],  # Input: modulated symbols
            out_sig=[np.complex64]  # Output: symbols with pilots inserted
        )
        
        self.frame_size = frame_size
        self.pilots_per_frame = pilots_per_frame
        self.pilot_spacing = pilot_spacing
        
        # Generate known pilot sequence (BPSK for robustness)
        self.pilot_sequence = self._generate_pilot_sequence(pilots_per_frame)
        self.pilot_symbol = complex(1.0, 0.0)  # BPSK pilot
        
        # State tracking
        self.symbol_count = 0
        self.frame_count = 0
        self.mcs_change_pending = False
        
        # Message ports
        self.message_port_register_in(pmt.intern("mcs_change"))
        self.set_msg_handler(pmt.intern("mcs_change"), self.handle_mcs_change)
        
        self.message_port_register_out(pmt.intern("pilot_info"))
        
        print(f"[PilotInserter] Initialized:")
        print(f"  - Frame size: {frame_size} symbols")
        print(f"  - Pilots per frame: {pilots_per_frame}")
        print(f"  - Pilot spacing: every {pilot_spacing} symbols")
        
        # Calculate overhead
        total_pilots = pilots_per_frame + (frame_size // pilot_spacing)
        overhead_pct = (total_pilots / (frame_size + total_pilots)) * 100
        print(f"  - Pilot overhead: {overhead_pct:.1f}%")
    
    def _generate_pilot_sequence(self, length):
        """
        Generate known pilot sequence for channel estimation.
        Uses alternating BPSK pattern for simplicity.
        """
        sequence = []
        for i in range(length):
            if i % 2 == 0:
                sequence.append(complex(1.0, 0.0))
            else:
                sequence.append(complex(-1.0, 0.0))
        return np.array(sequence, dtype=np.complex64)
    
    def handle_mcs_change(self, msg):
        """Handle message indicating MCS is about to change"""
        self.mcs_change_pending = True
        
        # Publish info about pilot burst
        info = pmt.make_dict()
        info = pmt.dict_add(info, pmt.intern("event"), pmt.intern("mcs_change_pilots"))
        info = pmt.dict_add(info, pmt.intern("num_pilots"), 
                           pmt.from_long(self.pilots_per_frame))
        self.message_port_pub(pmt.intern("pilot_info"), info)
    
    def insert_frame_header_pilots(self):
        """Insert pilots at start of frame"""
        self.frame_count += 1
        
        # Publish frame start info
        info = pmt.make_dict()
        info = pmt.dict_add(info, pmt.intern("event"), pmt.intern("frame_start"))
        info = pmt.dict_add(info, pmt.intern("frame_number"), 
                           pmt.from_long(self.frame_count))
        info = pmt.dict_add(info, pmt.intern("num_pilots"), 
                           pmt.from_long(self.pilots_per_frame))
        self.message_port_pub(pmt.intern("pilot_info"), info)
        
        return self.pilot_sequence.copy()
    
    def insert_mcs_change_pilots(self):
        """Insert pilots before MCS change"""
        self.mcs_change_pending = False
        return self.pilot_sequence.copy()
    
    def work(self, input_items, output_items):
        """
        GNU Radio work function.
        Inserts pilots into the symbol stream.
        """
        in0 = input_items[0]
        out = output_items[0]
        
        nin = len(in0)
        nout = len(out)
        
        produced = 0
        consumed = 0
        
        while consumed < nin and produced < nout:
            # Check if we need to insert frame header pilots
            if self.symbol_count == 0:
                pilots = self.insert_frame_header_pilots()
                n_pilots = len(pilots)
                
                if produced + n_pilots <= nout:
                    out[produced:produced + n_pilots] = pilots
                    produced += n_pilots
                    self.symbol_count += 1
                else:
                    break  # Not enough output space
            
            # Check if we need to insert MCS change pilots
            elif self.mcs_change_pending:
                pilots = self.insert_mcs_change_pilots()
                n_pilots = len(pilots)
                
                if produced + n_pilots <= nout:
                    out[produced:produced + n_pilots] = pilots
                    produced += n_pilots
                else:
                    break
            
            # Check if we need to insert periodic pilots
            elif self.symbol_count % self.pilot_spacing == 0:
                if produced < nout:
                    out[produced] = self.pilot_symbol
                    produced += 1
                    self.symbol_count += 1
                else:
                    break
            
            # Copy data symbols
            else:
                if consumed < nin and produced < nout:
                    out[produced] = in0[consumed]
                    produced += 1
                    consumed += 1
                    self.symbol_count += 1
                else:
                    break
            
            # Check for frame boundary
            if self.symbol_count >= self.frame_size:
                self.symbol_count = 0
        
        self.consume(0, consumed)
        return produced


class PilotRemover(gr.sync_block):
    """
    Removes pilot symbols from received stream.
    Also performs channel estimation using the pilots.
    """
    
    def __init__(self,
                 frame_size=1024,
                 pilots_per_frame=64,
                 pilot_spacing=16):
        """
        Initialize pilot remover.
        
        Args:
            frame_size: Size of data frame in symbols
            pilots_per_frame: Number of pilot symbols at frame start
            pilot_spacing: Pilot inserted every N data symbols
        """
        gr.sync_block.__init__(
            self,
            name="pilot_remover",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        
        self.frame_size = frame_size
        self.pilots_per_frame = pilots_per_frame
        self.pilot_spacing = pilot_spacing
        
        # Known pilot sequence (same as inserter)
        self.pilot_sequence = self._generate_pilot_sequence(pilots_per_frame)
        
        # State
        self.symbol_count = 0
        self.frame_count = 0
        
        # Channel estimate
        self.channel_estimate = complex(1.0, 0.0)
        
        # Message ports
        self.message_port_register_out(pmt.intern("channel_estimate"))
        
        print(f"[PilotRemover] Initialized")
    
    def _generate_pilot_sequence(self, length):
        """Generate same pilot sequence as inserter"""
        sequence = []
        for i in range(length):
            if i % 2 == 0:
                sequence.append(complex(1.0, 0.0))
            else:
                sequence.append(complex(-1.0, 0.0))
        return np.array(sequence, dtype=np.complex64)
    
    def estimate_channel(self, received_pilots):
        """
        Estimate channel from received pilot symbols.
        
        Args:
            received_pilots: Array of received pilot symbols
            
        Returns:
            Channel estimate (complex scalar)
        """
        # Simple LS (Least Squares) channel estimation
        # H = sum(Y * conj(X)) / sum(|X|^2)
        if len(received_pilots) != len(self.pilot_sequence):
            return self.channel_estimate
        
        numerator = np.sum(received_pilots * np.conj(self.pilot_sequence))
        denominator = np.sum(np.abs(self.pilot_sequence)**2)
        
        if denominator > 0:
            self.channel_estimate = numerator / denominator
        
        # Publish channel estimate
        snr_estimate = 10 * np.log10(np.abs(self.channel_estimate)**2 + 1e-10)
        
        msg = pmt.make_dict()
        msg = pmt.dict_add(msg, pmt.intern("channel_magnitude"), 
                          pmt.from_double(float(np.abs(self.channel_estimate))))
        msg = pmt.dict_add(msg, pmt.intern("channel_phase"), 
                          pmt.from_double(float(np.angle(self.channel_estimate))))
        msg = pmt.dict_add(msg, pmt.intern("snr_estimate_db"), 
                          pmt.from_double(float(snr_estimate)))
        self.message_port_pub(pmt.intern("channel_estimate"), msg)
        
        return self.channel_estimate
    
    def work(self, input_items, output_items):
        """Remove pilots and estimate channel"""
        in0 = input_items[0]
        out = output_items[0]
        
        nin = len(in0)
        nout = len(out)
        
        produced = 0
        consumed = 0
        
        pilot_buffer = []
        
        while consumed < nin and produced < nout:
            # Collect frame header pilots
            if self.symbol_count == 0:
                while len(pilot_buffer) < self.pilots_per_frame and consumed < nin:
                    pilot_buffer.append(in0[consumed])
                    consumed += 1
                
                if len(pilot_buffer) == self.pilots_per_frame:
                    # Estimate channel
                    self.estimate_channel(np.array(pilot_buffer))
                    pilot_buffer = []
                    self.symbol_count += 1
                    self.frame_count += 1
                else:
                    break
            
            # Skip periodic pilots
            elif self.symbol_count % self.pilot_spacing == 0:
                if consumed < nin:
                    # Skip pilot (could also use for tracking)
                    consumed += 1
                    self.symbol_count += 1
                else:
                    break
            
            # Copy data symbols
            else:
                if consumed < nin and produced < nout:
                    out[produced] = in0[consumed]
                    produced += 1
                    consumed += 1
                    self.symbol_count += 1
                else:
                    break
            
            # Frame boundary
            if self.symbol_count >= self.frame_size:
                self.symbol_count = 0
        
        self.consume(0, consumed)
        return produced
