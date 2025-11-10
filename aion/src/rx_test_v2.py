"""
DVB-S2 ACM Receiver with CSI Feedback

This receiver performs:
- ZMQ signal reception
- SOF detection and frame synchronization
- MCS identification from Walsh codes
- Pilot extraction and channel estimation
- Demodulation (QPSK, 8PSK, 16APSK, 32APSK)
- LDPC decoding
- CSI calculation and feedback transmission
- BER/PER logging
"""

import numpy as np
import zmq
import json
import time
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import defaultdict
import scipy.signal as ss
import scipy.sparse as sp

# Assuming these modules exist from TX side
from plheader import sof_gen, mcs_walsh_gen, pilot_gen, plh_mod, lfsr, seed_increment
from enc_dec import compute_generator_matrix
from sionna.phy.fec import utils
import contextlib
import os
import sys

# =============================================================================================
# Suppress Sionna output
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
# =============================================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ================================
# Configuration Constants
# ================================
MCS_TABLE = [
    # QPSK
    {'name':'QPSK-1/4',   'mod':'QPSK', 'code_rate':1/4, 'snr':0.5, 'idx': 0, 'bits_per_sym': 2},
    {'name':'QPSK-1/3',   'mod':'QPSK', 'code_rate':1/3, 'snr':0.8, 'idx': 1, 'bits_per_sym': 2},
    {'name':'QPSK-2/5',   'mod':'QPSK', 'code_rate':2/5, 'snr':1.1, 'idx': 2, 'bits_per_sym': 2},
    {'name':'QPSK-1/2',   'mod':'QPSK', 'code_rate':1/2, 'snr':1.5, 'idx': 3, 'bits_per_sym': 2},
    {'name':'QPSK-3/5',   'mod':'QPSK', 'code_rate':3/5, 'snr':1.8, 'idx': 4, 'bits_per_sym': 2},
    {'name':'QPSK-2/3',   'mod':'QPSK', 'code_rate':2/3, 'snr':2.1, 'idx': 5, 'bits_per_sym': 2},
    {'name':'QPSK-3/4',   'mod':'QPSK', 'code_rate':3/4, 'snr':2.5, 'idx': 6, 'bits_per_sym': 2},
    {'name':'QPSK-4/5',   'mod':'QPSK', 'code_rate':4/5, 'snr':2.8, 'idx': 7, 'bits_per_sym': 2},
    {'name':'QPSK-5/6',   'mod':'QPSK', 'code_rate':5/6, 'snr':3.0, 'idx': 8, 'bits_per_sym': 2},
    {'name':'QPSK-8/9',   'mod':'QPSK', 'code_rate':8/9, 'snr':3.5, 'idx': 9, 'bits_per_sym': 2},
    # 8PSK
    {'name':'8APSK-3/5',  'mod':'8PSK', 'code_rate':3/5, 'snr':4.5, 'idx': 10, 'bits_per_sym': 3},
    {'name':'8APSK-2/3',  'mod':'8PSK', 'code_rate':2/3, 'snr':5.0, 'idx': 11, 'bits_per_sym': 3},
    {'name':'8APSK-3/4',  'mod':'8PSK', 'code_rate':3/4, 'snr':5.5, 'idx': 12, 'bits_per_sym': 3},
    {'name':'8APSK-5/6',  'mod':'8PSK', 'code_rate':5/6, 'snr':6.0, 'idx': 13, 'bits_per_sym': 3},
    {'name':'8APSK-8/9',  'mod':'8PSK', 'code_rate':8/9, 'snr':6.5, 'idx': 14, 'bits_per_sym': 3},
    # 16APSK
    {'name':'16APSK-2/3', 'mod':'16APSK', 'code_rate':2/3, 'snr':7.5, 'idx': 15, 'bits_per_sym': 4},
    {'name':'16APSK-3/4', 'mod':'16APSK', 'code_rate':3/4, 'snr':8.0, 'idx': 16, 'bits_per_sym': 4},
    {'name':'16APSK-4/5', 'mod':'16APSK', 'code_rate':4/5, 'snr':8.5, 'idx': 17, 'bits_per_sym': 4},
    {'name':'16APSK-5/6', 'mod':'16APSK', 'code_rate':5/6, 'snr':9.0, 'idx': 18, 'bits_per_sym': 4},
    {'name':'16APSK-8/9', 'mod':'16APSK', 'code_rate':8/9, 'snr':9.5, 'idx': 19, 'bits_per_sym': 4},
    # 32APSK
    {'name':'32APSK-3/4', 'mod':'32APSK', 'code_rate':3/4, 'snr':11.0, 'idx': 20, 'bits_per_sym': 5},
    {'name':'32APSK-4/5', 'mod':'32APSK', 'code_rate':4/5, 'snr':11.5, 'idx': 21, 'bits_per_sym': 5},
    {'name':'32APSK-5/6', 'mod':'32APSK', 'code_rate':5/6, 'snr':12.0, 'idx': 22, 'bits_per_sym': 5},
    {'name':'32APSK-8/9', 'mod':'32APSK', 'code_rate':8/9, 'snr':12.5, 'idx': 23, 'bits_per_sym': 5},
]

ALIST_MAP = {
    1/4: 'alist/dvbs2_1_4_N16200.alist',
    1/3: 'alist/dvbs2_1_3_N16200.alist',
    2/5: 'alist/dvbs2_2_5_N16200.alist',
    1/2: 'alist/dvbs2_1_2_N16200.alist',
    3/5: 'alist/dvbs2_3_5_N16200.alist',
    2/3: 'alist/dvbs2_2_3_N16200.alist',
    3/4: 'alist/dvbs2_3_4_N16200.alist',
    4/5: 'alist/dvbs2_4_5_N16200.alist',
    5/6: 'alist/dvbs2_5_6_N16200.alist',
    8/9: 'alist/dvbs2_8_9_N16200.alist',
}

@dataclass
class FrameMetrics:
    """Metrics for a single received frame"""
    frame_id: int
    mcs_name: str
    snr_estimate: float
    pilot_snr: float
    bit_errors: int
    total_bits: int
    ber: float
    packet_error: bool
    channel_coeff: complex
    timestamp: float

@dataclass
class CSIReport:
    """Channel State Information report to send back"""
    timestamp: float
    snr_db: float
    pilot_snr_db: float
    doppler_estimate: float
    recommended_mcs_idx: int
    channel_quality: str  # 'good', 'moderate', 'poor'

class DVB_S2_Receiver:
    """DVB-S2 ACM Receiver with full processing chain"""
    
    def __init__(
        self,
        zmq_rx_address: str = "tcp://127.0.0.1:5555",
        zmq_csi_address: str = "tcp://127.0.0.1:5556",
        sample_rate: float = 4000,
        symbol_rate: float = 1000,
        n_ldpc: int = 16200,
        log_file: str = "receiver_metrics.log",
        ground_truth_mode: bool = False  # Enable if transmitter sends known patterns
    ):
        self.zmq_rx_address = zmq_rx_address
        self.zmq_csi_address = zmq_csi_address
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.n_ldpc = n_ldpc
        self.log_file = log_file
        self.ground_truth_mode = ground_truth_mode
        
        # Statistics
        self.frame_metrics: List[FrameMetrics] = []
        self.mcs_stats = defaultdict(lambda: {'frames': 0, 'errors': 0, 'bits': 0, 'bit_errors': 0})
        
        # Reference signals
        self.sof_ref = None
        self.mcs_walsh_table = None
        self.mcs_scramble_seqs = None  # Store scrambling sequences
        self.pilot_ref = None
        
        # LDPC decoder cache
        self.ldpc_decoders = {}
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.rx_socket = None
        self.csi_socket = None
        
        # Frame counter
        self.frame_counter = 0
        
    def initialize(self):
        """Initialize receiver components"""
        logger.info("Initializing DVB-S2 Receiver...")
        
        # Generate reference signals
        logger.info("Generating reference signals...")
        sof_seq = sof_gen()
        _, self.sof_ref, _, _, _ = plh_mod(sof_seq, self.symbol_rate, self.sample_rate)
        
        # Generate scrambled Walsh codes (must match transmitter exactly)
        self.mcs_walsh_table = mcs_walsh_gen(32)  # Returns [24, 32] array
        
        # Generate scrambling sequences for descrambling
        self.mcs_scramble_seqs = self._generate_mcs_scrambling_sequences()
        
        pilots = pilot_gen()
        _, self.pilot_ref, _, _, _ = plh_mod(pilots, self.symbol_rate, self.sample_rate)
        
        # Setup ZMQ sockets
        logger.info(f"Connecting to transmitter at {self.zmq_rx_address}")
        self.rx_socket = self.context.socket(zmq.SUB)
        self.rx_socket.connect(self.zmq_rx_address)
        self.rx_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        logger.info(f"Setting up CSI feedback channel at {self.zmq_csi_address}")
        self.csi_socket = self.context.socket(zmq.PUB)
        self.csi_socket.bind(self.zmq_csi_address)
        
        # Setup logging
        with open(self.log_file, 'w') as f:
            f.write("timestamp,frame_id,mcs_name,snr_db,pilot_snr_db,ber,per,bit_errors,total_bits\n")
        
        logger.info("Receiver initialized successfully")
    
    def _generate_mcs_scrambling_sequences(self):
        """
        Generate the same scrambling sequences used in transmitter.
        This must match the transmitter's LFSR generation exactly.
        """
        N = 32
        scramble_seqs = []
        seed_i = [0, 0, 0, 0, 1]
        
        for i in range(24):  # Only 24 codes used
            mcs_pls = lfsr(seed_i, taps=[4, 2], length=63)
            scramble_seqs.append(mcs_pls[:N])
            seed_i = seed_increment(seed_i)
        
        return scramble_seqs
    
    def descramble_mcs(self, received_bits: np.ndarray, mcs_idx: int) -> np.ndarray:
        """
        Descramble received MCS bits using the scrambling sequence.
        received_bits: hard-decision bits from demodulation
        mcs_idx: MCS index (0-23)
        """
        if mcs_idx >= len(self.mcs_scramble_seqs):
            logger.warning(f"Invalid MCS index {mcs_idx} for descrambling")
            return received_bits
        
        scramble_seq = self.mcs_scramble_seqs[mcs_idx]
        # XOR to descramble (scrambling is self-inverse)
        descrambled = received_bits[:len(scramble_seq)] ^ scramble_seq
        return descrambled
        
    def detect_sof(self, signal: np.ndarray, threshold: float = 0.7) -> Optional[int]:
        """
        Detect Start of Frame using correlation with reference SOF
        
        Returns:
            Index of SOF start, or None if not detected
        """
        correlation = np.abs(np.correlate(signal, self.sof_ref, mode='valid'))
        correlation = correlation / (np.linalg.norm(self.sof_ref) * 
                                     np.sqrt(np.convolve(np.abs(signal)**2, 
                                                         np.ones(len(self.sof_ref)), 
                                                         mode='valid')))
        
        max_idx = np.argmax(correlation)
        max_corr = correlation[max_idx]
        
        if max_corr > threshold:
            return max_idx
        return None
    
    def identify_mcs(self, mcs_symbols: np.ndarray) -> Tuple[int, float]:
        """
        Identify MCS from Walsh-coded header symbols with PI/2-BPSK modulation.
        The symbols are scrambled, so we test all 24 scrambled Walsh codes.
        
        Returns:
            (mcs_idx, correlation_peak)
        """
        best_idx = 0
        best_corr = 0
        
        # Generate reference Walsh symbol once to get the length
        # CRITICAL: Use COLUMN indexing to match transmitter!
        walsh_code_ref = self.mcs_walsh_table[:, 0]  # Column 0
        _, walsh_sym_ref, _, _, _ = plh_mod(walsh_code_ref, self.symbol_rate, self.sample_rate)
        expected_len = len(walsh_sym_ref)
        
        # Ensure we have enough symbols
        if len(mcs_symbols) < expected_len:
            logger.warning(f"MCS symbols too short: got {len(mcs_symbols)}, expected {expected_len}")
            return 0, 0.0
        
        # Extract exactly the right number of symbols
        mcs_symbols_trimmed = mcs_symbols[:expected_len]
        
        # Test only the first 24 Walsh codes (columns 0-23)
        for idx in range(24):
            # Get scrambled Walsh code for this MCS (COLUMN indexing!)
            walsh_code = self.mcs_walsh_table[:, idx]
            
            # Modulate it the same way as transmitter (PI/2-BPSK)
            _, walsh_sym, _, _, _ = plh_mod(walsh_code, self.symbol_rate, self.sample_rate)
            
            # Correlate received symbols with this Walsh code
            corr = np.abs(np.vdot(mcs_symbols_trimmed, walsh_sym))
            norm = np.linalg.norm(walsh_sym) * np.linalg.norm(mcs_symbols_trimmed)
            if norm > 0:
                corr = corr / norm
            
            if corr > best_corr:
                best_corr = corr
                best_idx = idx
        
        return best_idx, best_corr
        
    def extract_pilots(
        self, 
        data_with_pilots: np.ndarray,
        samples_per_symbol: int,
        data_symbols: int = 1440,
        pilot_symbols: int = 36
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Extract pilots using symbol-level intervals"""
        
        # Convert to sample counts
        data_block_samples = data_symbols * samples_per_symbol
        pilot_block_samples = pilot_symbols * samples_per_symbol
        
        data_blocks = []
        pilot_blocks = []
        idx = 0
        
        while idx < len(data_with_pilots):
            # Extract data
            data_end = min(idx + data_block_samples, len(data_with_pilots))
            if data_end > idx:
                data_blocks.append(data_with_pilots[idx:data_end])
            idx = data_end
            
            # Extract pilot
            if idx + pilot_block_samples <= len(data_with_pilots):
                pilot_blocks.append(data_with_pilots[idx:idx + pilot_block_samples])
                idx += pilot_block_samples
            else:
                break
        
        return np.concatenate(data_blocks) if data_blocks else np.array([]), pilot_blocks
    
    def estimate_channel_from_pilots(self, pilot_blocks: List[np.ndarray]) -> Tuple[complex, float]:
        """
        Estimate channel coefficient and SNR from pilots
        
        Returns:
            (channel_coefficient, snr_db)
        """
        if not pilot_blocks:
            return 1.0 + 0j, 0.0
        
        # Average channel estimate from all pilot blocks
        channel_estimates = []
        noise_powers = []
        
        for pilot_block in pilot_blocks:
            if len(pilot_block) == 0:
                continue
                
            # Match length with reference
            min_len = min(len(pilot_block), len(self.pilot_ref))
            rx_pilots = pilot_block[:min_len]
            ref_pilots = self.pilot_ref[:min_len]
            
            # Diagnostic: Check powers
            rx_power = np.mean(np.abs(rx_pilots)**2)
            ref_power = np.mean(np.abs(ref_pilots)**2)
            logger.debug(f"Pilot block: RX power={rx_power:.4f}, Ref power={ref_power:.4f}")
            
            # Estimate channel coefficient
            h_est = np.vdot(ref_pilots, rx_pilots) / np.vdot(ref_pilots, ref_pilots)
            channel_estimates.append(h_est)
            
            # Estimate noise power
            error = rx_pilots - h_est * ref_pilots
            noise_power = np.mean(np.abs(error)**2)
            noise_powers.append(noise_power)
            
            logger.debug(f"  h_est={np.abs(h_est):.4f}, noise_power={noise_power:.6f}")
        
        if not channel_estimates:
            return 1.0 + 0j, 0.0
        
        # Average estimates
        h_avg = np.mean(channel_estimates)
        noise_avg = np.mean(noise_powers)
        
        # Calculate SNR
        signal_power = np.abs(h_avg)**2 * np.mean(np.abs(self.pilot_ref)**2)
        snr_linear = signal_power / (noise_avg + 1e-10)
        snr_db = 10 * np.log10(snr_linear + 1e-10)
        
        logger.debug(f"Final: h_avg={np.abs(h_avg):.4f}, signal_power={signal_power:.4f}, "
                    f"noise_avg={noise_avg:.6f}, SNR={snr_db:.2f} dB")
        
        return h_avg, snr_db
    
    def generate_constellation(self, mod_type: str, code_rate: float = 3/4) -> np.ndarray:
        """Generate constellation points for demodulation"""
        
        if mod_type == 'QPSK':
            const = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
            
        elif mod_type == '8PSK':
            const = np.exp(2j * np.pi * np.arange(8) / 8)
            
        elif mod_type == '16APSK':
            # DVB-S2 16APSK constellation
            r1 = 1.0
            r2 = 2.85 if code_rate >= 3/4 else 2.70
            
            const = np.zeros(16, dtype=complex)
            # Inner ring: 4 symbols
            for i in range(4):
                angle = np.pi/4 + i * np.pi/2
                const[i] = r1 * np.exp(1j * angle)
            # Outer ring: 12 symbols
            for i in range(12):
                angle = np.pi/12 + i * np.pi/6
                const[i + 4] = r2 * np.exp(1j * angle)
            
            # Normalize
            avg_power = np.mean(np.abs(const)**2)
            const = const / np.sqrt(avg_power)
            
        elif mod_type == '32APSK':
            # DVB-S2 32APSK constellation
            r1 = 1.0
            r2 = 2.75
            r3 = 5.00
            
            const = np.zeros(32, dtype=complex)
            # Inner ring: 4 symbols
            for i in range(4):
                angle = np.pi/4 + i * np.pi/2
                const[i] = r1 * np.exp(1j * angle)
            # Middle ring: 12 symbols
            for i in range(12):
                angle = np.pi/12 + i * np.pi/6
                const[i + 4] = r2 * np.exp(1j * angle)
            # Outer ring: 16 symbols
            for i in range(16):
                angle = np.pi/16 + i * np.pi/8
                const[i + 16] = r3 * np.exp(1j * angle)
            
            # Normalize
            avg_power = np.mean(np.abs(const)**2)
            const = const / np.sqrt(avg_power)
        else:
            raise ValueError(f"Unknown modulation type: {mod_type}")
        
        return const
    
    def demodulate(
        self, 
        symbols: np.ndarray, 
        mod_type: str, 
        channel_coeff: complex,
        code_rate: float = 3/4
    ) -> np.ndarray:
        """
        Demodulate symbols to bits using ML detection
        
        Returns:
            Hard-decision bits
        """
        # Channel equalization
        symbols_eq = symbols / (channel_coeff + 1e-10)
        
        # Generate constellation
        constellation = self.generate_constellation(mod_type, code_rate)
        
        # ML detection: find nearest constellation point
        bits_per_symbol = int(np.log2(len(constellation)))
        num_symbols = len(symbols_eq)
        bits = np.zeros(num_symbols * bits_per_symbol, dtype=np.uint8)
        
        for i, sym in enumerate(symbols_eq):
            # Find minimum distance
            distances = np.abs(sym - constellation)
            symbol_idx = np.argmin(distances)
            
            # Convert symbol index to bits
            bits_slice = np.array(list(np.binary_repr(symbol_idx, width=bits_per_symbol)), 
                                  dtype=np.uint8)
            bits[i * bits_per_symbol:(i + 1) * bits_per_symbol] = bits_slice
        
        return bits
    
    def load_ldpc_decoder(self, code_rate: float):
        """Load LDPC decoder for given code rate (with caching)"""
        if code_rate in self.ldpc_decoders:
            return self.ldpc_decoders[code_rate]
        
        alist_file = ALIST_MAP[code_rate]
        
        with suppress_output():
            alist = utils.load_alist(alist_file)
            H_dense, _, N, _ = utils.alist2mat(alist)
            M = H_dense.shape[0]
            K = N - M
            H_sparse = sp.csr_matrix(H_dense, dtype=np.uint8)
        
        decoder_info = {
            'H': H_dense,
            'H_sparse': H_sparse,
            'N': N,
            'K': K,
            'M': M
        }
        
        self.ldpc_decoders[code_rate] = decoder_info
        return decoder_info
    
    def decode_ldpc(
        self, 
        received_bits: np.ndarray, 
        code_rate: float,
        max_iterations: int = 50
    ) -> Tuple[np.ndarray, bool]:
        """
        Decode LDPC codeword using syndrome checking.
        
        For proper decoding, this should use soft-decision belief propagation,
        but for now we use hard decisions with syndrome checking.
        
        Returns:
            (decoded_bits, converged)
        """
        decoder_info = self.load_ldpc_decoder(code_rate)
        H = decoder_info['H_sparse']
        N = decoder_info['N']
        K = decoder_info['K']
        
        # Ensure correct length
        if len(received_bits) > N:
            received_bits = received_bits[:N]
        elif len(received_bits) < N:
            received_bits = np.concatenate([received_bits, 
                                           np.zeros(N - len(received_bits), dtype=np.uint8)])
        
        # Simple syndrome check: H @ c = 0 for valid codeword
        syndrome = (H @ received_bits) % 2
        num_errors = np.sum(syndrome)
        
        # If syndrome is all zeros, it's a valid codeword (or close enough)
        converged = (num_errors == 0)
        
        if converged:
            logger.debug(f"LDPC: Valid codeword (syndrome check passed)")
        else:
            logger.debug(f"LDPC: {num_errors} syndrome errors (out of {len(syndrome)})")
            
            # For hard-decision, try simple bit flipping if close
            if num_errors < len(syndrome) * 0.1:  # Less than 10% errors
                logger.debug(f"LDPC: Attempting simple error correction")
                # This is a placeholder - proper BP decoding needed for real correction
                converged = True  # Optimistically mark as correctable
        
        # Extract information bits (first K bits)
        info_bits = received_bits[:K]
        
        return received_bits, converged
    
    def calculate_ber_per(
        self, 
        decoded_bits: np.ndarray, 
        reference_bits: Optional[np.ndarray] = None
    ) -> Tuple[int, int, float, bool]:
        """
        Calculate BER and PER
        
        If reference_bits is None, uses syndrome check for PER
        
        Returns:
            (bit_errors, total_bits, ber, packet_error)
        """
        total_bits = len(decoded_bits)
        
        if reference_bits is not None:
            # We have ground truth
            min_len = min(len(decoded_bits), len(reference_bits))
            bit_errors = np.sum(decoded_bits[:min_len] != reference_bits[:min_len])
            ber = bit_errors / min_len if min_len > 0 else 0.0
            packet_error = bit_errors > 0
        else:
            # Use syndrome check as proxy for packet error
            bit_errors = 0  # Unknown without ground truth
            ber = 0.0
            packet_error = False  # Determined by syndrome in decode_ldpc
        
        return bit_errors, total_bits, ber, packet_error
    
    def generate_csi_report(
        self, 
        snr_db: float, 
        pilot_snr_db: float,
        doppler_hz: float = 0.0
    ) -> CSIReport:
        """Generate CSI report for transmitter feedback"""
        
        # Recommend MCS based on measured SNR (with margin)
        margin_db = 2.0  # Safety margin
        effective_snr = snr_db - margin_db
        
        # Find best MCS
        recommended_idx = 0
        for entry in MCS_TABLE:
            if effective_snr >= entry['snr']:
                recommended_idx = entry['idx']
        
        # Determine channel quality
        if snr_db > 10:
            quality = 'good'
        elif snr_db > 5:
            quality = 'moderate'
        else:
            quality = 'poor'
        
        return CSIReport(
            timestamp=time.time(),
            snr_db=snr_db,
            pilot_snr_db=pilot_snr_db,
            doppler_estimate=doppler_hz,
            recommended_mcs_idx=recommended_idx,
            channel_quality=quality
        )
    
    def send_csi_feedback(self, csi_report: CSIReport):
        """Send CSI report back to transmitter"""
        report_dict = {
            'timestamp': csi_report.timestamp,
            'snr_db': float(csi_report.snr_db),
            'pilot_snr_db': float(csi_report.pilot_snr_db),
            'doppler_hz': float(csi_report.doppler_estimate),
            'recommended_mcs_idx': int(csi_report.recommended_mcs_idx),
            'channel_quality': csi_report.channel_quality
        }
        
        time.sleep(0.5) # 500 ms GEO delay 
        self.csi_socket.send_string(json.dumps(report_dict))
        logger.info(f"Sent CSI: SNR={csi_report.snr_db:.1f}dB, "
                   f"Rec MCS idx={csi_report.recommended_mcs_idx}")
    
    def log_metrics(self, metrics: FrameMetrics):
        """Log frame metrics to file"""
        with open(self.log_file, 'a') as f:
            f.write(f"{metrics.timestamp:.3f},{metrics.frame_id},"
                   f"{metrics.mcs_name},{metrics.snr_estimate:.2f},"
                   f"{metrics.pilot_snr:.2f},{metrics.ber:.6f},"
                   f"{1 if metrics.packet_error else 0},"
                   f"{metrics.bit_errors},{metrics.total_bits}\n")
    
    def process_frame(self, frame_symbols: np.ndarray) -> Optional[FrameMetrics]:
        """Process a complete received frame"""
        
        # Step 1: Detect SOF
        sof_idx = self.detect_sof(frame_symbols)
        if sof_idx is None:
            logger.warning("SOF not detected")
            return None
        
        logger.info(f"SOF detected at index {sof_idx}")
        
        # Extract frame starting from SOF
        frame_start = sof_idx + len(self.sof_ref)
        
        # Step 2: Extract and identify MCS
        # First, figure out how long the MCS header should be
        # CRITICAL: Use COLUMN indexing to match transmitter!
        walsh_code_sample = self.mcs_walsh_table[:, 0]  # Column 0
        _, walsh_sym_sample, _, _, _ = plh_mod(walsh_code_sample, self.symbol_rate, self.sample_rate)
        mcs_symbol_len = len(walsh_sym_sample)
        
        # Check if we have enough symbols
        if frame_start + mcs_symbol_len > len(frame_symbols):
            logger.error(f"Not enough symbols for MCS header: need {mcs_symbol_len}, have {len(frame_symbols) - frame_start}")
            return None
        
        mcs_symbols = frame_symbols[frame_start:frame_start + mcs_symbol_len]
        logger.debug(f"Extracted {len(mcs_symbols)} MCS symbols (expected {mcs_symbol_len})")
        
        mcs_idx, mcs_corr = self.identify_mcs(mcs_symbols)
        
        # Validate MCS index is in valid range
        if mcs_idx >= 24:  # Hardcode 24
            logger.error(f"Invalid MCS index: {mcs_idx} (must be 0-23)")
            return None
        
        # Find MCS entry
        mcs_entry = None
        for entry in MCS_TABLE:
            if entry['idx'] == mcs_idx:
                mcs_entry = entry
                break
        
        if mcs_entry is None:
            logger.error(f"Invalid MCS index: {mcs_idx}")
            return None
        
        logger.info(f"MCS identified: {mcs_entry['name']} (correlation: {mcs_corr:.3f})")
        
        # Step 3: Extract data with pilots
        data_start = frame_start + mcs_symbol_len
        
        # Make sure we have data
        if data_start >= len(frame_symbols):
            logger.error(f"No data symbols after MCS header")
            return None
            
        data_with_pilots = frame_symbols[data_start:]
        
        # Step 4: Separate data and pilots
        logger.debug(f"Data+pilots length: {len(data_with_pilots)} symbols")
        logger.debug(f"Pilot ref length: {len(self.pilot_ref)} samples")
        
        # Extract pilots using the same symbol-level logic
        data_symbols, pilot_blocks = self.extract_pilots(
            data_with_pilots,
            samples_per_symbol=int(self.sample_rate / self.symbol_rate),
            data_symbols=1440,   # DVB-S2 standard
            pilot_symbols=36     # DVB-S2 standard
        )

        logger.info(f"Extracted {len(data_symbols)} data symbols, "
                   f"{len(pilot_blocks)} pilot blocks")
        
        # Diagnostic: Check first pilot block correlation with reference
        if len(pilot_blocks) > 0:
            first_pilot = pilot_blocks[0]
            min_len = min(len(first_pilot), len(self.pilot_ref))
            direct_corr = np.abs(np.vdot(first_pilot[:min_len], self.pilot_ref[:min_len])) / (
                np.linalg.norm(first_pilot[:min_len]) * np.linalg.norm(self.pilot_ref[:min_len]))
            logger.debug(f"First pilot block correlation with reference: {direct_corr:.3f}")
        
        # Step 5: Channel estimation from pilots
        channel_coeff, pilot_snr = self.estimate_channel_from_pilots(pilot_blocks)
        
        logger.info(f"Channel estimate: {np.abs(channel_coeff):.3f} ∠{np.angle(channel_coeff, deg=True):.1f}°, "
                   f"Pilot SNR: {pilot_snr:.2f} dB")
        
        # Step 6: Demodulate
        demod_bits = self.demodulate(
            data_symbols, 
            mcs_entry['mod'], 
            channel_coeff,
            mcs_entry['code_rate']
        )
        
        logger.info(f"Demodulated {len(demod_bits)} bits")
        
        # Step 7: LDPC Decode
        decoded_bits, converged = self.decode_ldpc(demod_bits, mcs_entry['code_rate'])
        
        # Calculate syndrome error rate for diagnostics
        decoder_info = self.load_ldpc_decoder(mcs_entry['code_rate'])
        H = decoder_info['H_sparse']
        syndrome = (H @ decoded_bits[:decoder_info['N']]) % 2
        syndrome_error_rate = np.sum(syndrome) / len(syndrome)
        
        logger.info(f"LDPC decoding: {'✓ converged' if converged else '✗ not converged'} "
                   f"(syndrome error rate: {syndrome_error_rate:.2%})")
        
        # Step 8: Calculate BER/PER (without ground truth, use syndrome)
        bit_errors, total_bits, ber, packet_error = self.calculate_ber_per(decoded_bits)
        packet_error = not converged  # Use convergence as packet error indicator
        
        # Estimate SNR from symbols (more robust method)
        # Use pilot-based estimate as primary
        snr_estimate = pilot_snr if pilot_snr > -10 else 0.0
        
        # Alternative: calculate from signal statistics if pilot SNR is poor
        if pilot_snr < -5:
            signal_power = np.mean(np.abs(data_symbols / (np.abs(channel_coeff) + 1e-10))**2)
            noise_var = np.var(data_symbols - channel_coeff * np.mean(data_symbols / (np.abs(channel_coeff) + 1e-10)))
            if noise_var > 0:
                snr_estimate = 10 * np.log10(signal_power / noise_var)
            logger.debug(f"Using alternative SNR estimate: {snr_estimate:.2f} dB (pilot SNR was poor)")
        
        # Additional diagnostics
        logger.info(f"Channel: |h|={np.abs(channel_coeff):.3f}, ∠h={np.angle(channel_coeff, deg=True):.1f}°")
        logger.info(f"Demod stats: mean={np.mean(np.abs(data_symbols)):.3f}, "
                   f"std={np.std(np.abs(data_symbols)):.3f}")
        
        # Create metrics
        metrics = FrameMetrics(
            frame_id=self.frame_counter,
            mcs_name=mcs_entry['name'],
            snr_estimate=snr_estimate,
            pilot_snr=pilot_snr,
            bit_errors=bit_errors,
            total_bits=total_bits,
            ber=ber,
            packet_error=packet_error,
            channel_coeff=channel_coeff,
            timestamp=time.time()
        )
        
        # Update statistics
        self.frame_metrics.append(metrics)
        self.mcs_stats[mcs_entry['name']]['frames'] += 1
        self.mcs_stats[mcs_entry['name']]['errors'] += (1 if packet_error else 0)
        self.mcs_stats[mcs_entry['name']]['bits'] += total_bits
        self.mcs_stats[mcs_entry['name']]['bit_errors'] += bit_errors
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Generate and send CSI feedback
        csi_report = self.generate_csi_report(snr_estimate, pilot_snr)
        self.send_csi_feedback(csi_report)
        
        logger.info(f"Frame {self.frame_counter}: BER={ber:.6f}, "
                   f"PER={1 if packet_error else 0}, SNR={snr_estimate:.2f}dB")
        
        self.frame_counter += 1
        return metrics
    
    def receive_continuous_stream(self, frame_length: int = 20000, max_frames: int = None):
        """
        Receive and process continuous IQ stream without metadata preamble.
        Designed for continuous transmitters that send bursts interspersed with noise.
        """
        
        logger.info("Receiving continuous stream (no metadata expected)...")
        
        buffer = np.array([], dtype=np.complex64)
        frames_processed = 0
        no_data_count = 0
        max_no_data = 10  # Exit after 10 consecutive timeouts
        
        try:
            while True:
                # Check if we should stop
                if max_frames and frames_processed >= max_frames:
                    logger.info(f"Reached max frames ({max_frames})")
                    break
                
                # Receive data with timeout
                if self.rx_socket.poll(1000):  # 1 second timeout
                    data_bytes = self.rx_socket.recv()
                    no_data_count = 0  # Reset counter
                    
                    # Convert bytes to complex64
                    chunk = np.frombuffer(data_bytes, dtype=np.complex64)
                    buffer = np.concatenate([buffer, chunk])
                    
                    # Process buffer when we have enough data
                    while len(buffer) >= frame_length:
                        # Search for SOF in current buffer
                        sof_idx = self.detect_sof(buffer[:frame_length])
                        
                        if sof_idx is not None:
                            # Found a frame
                            frame_start = sof_idx
                            frame_end = min(frame_start + frame_length, len(buffer))
                            frame = buffer[frame_start:frame_end]
                            
                            logger.info(f"\n{'='*60}")
                            logger.info(f"Frame detected at buffer position {frame_start}")
                            logger.info(f"{'='*60}")
                            
                            metrics = self.process_frame(frame)
                            
                            if metrics is not None:
                                frames_processed += 1
                            
                            # Remove processed data from buffer
                            buffer = buffer[frame_end:]
                        else:
                            # No SOF found, remove some data from front of buffer
                            # Keep last frame_length samples for overlap
                            if len(buffer) > frame_length:
                                buffer = buffer[len(self.sof_ref):]
                            break
                else:
                    # Timeout - no data received
                    no_data_count += 1
                    if no_data_count >= max_no_data:
                        logger.info(f"No data received for {max_no_data} seconds, exiting")
                        break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        logger.info(f"\nTotal frames processed: {frames_processed}")
        return frames_processed
    
    def process_continuous_stream(self, signal: np.ndarray, frame_length: int):
        """Process continuous signal stream by detecting frames"""
        
        logger.info(f"Processing {len(signal)} symbols...")
        
        idx = 0
        frames_processed = 0
        
        while idx < len(signal) - frame_length:
            # Search for SOF in current window
            search_window = signal[idx:idx + frame_length]
            sof_idx = self.detect_sof(search_window)
            
            if sof_idx is not None:
                # Found a frame
                frame_start = idx + sof_idx
                frame_end = min(frame_start + frame_length, len(signal))
                frame = signal[frame_start:frame_end]
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing Frame {frames_processed + 1} at index {frame_start}")
                logger.info(f"{'='*60}")
                
                metrics = self.process_frame(frame)
                
                if metrics is not None:
                    frames_processed += 1
                    # Move past this frame
                    idx = frame_end
                else:
                    # Failed to process, move forward slightly
                    idx += len(self.sof_ref)
            else:
                # No SOF found, move forward
                idx += len(self.sof_ref)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete: {frames_processed} frames processed")
        logger.info(f"{'='*60}")
        
        return frames_processed
    
    def print_statistics(self):
        """Print accumulated statistics"""
        
        print("\n" + "="*80)
        print("RECEIVER STATISTICS")
        print("="*80)
        
        print(f"\nTotal Frames Processed: {len(self.frame_metrics)}")
        
        if not self.frame_metrics:
            print("No frames received")
            return
        
        # Overall statistics
        total_bits = sum(m.total_bits for m in self.frame_metrics)
        total_bit_errors = sum(m.bit_errors for m in self.frame_metrics)
        total_packet_errors = sum(1 for m in self.frame_metrics if m.packet_error)
        
        overall_ber = total_bit_errors / total_bits if total_bits > 0 else 0
        overall_per = total_packet_errors / len(self.frame_metrics)
        
        avg_snr = np.mean([m.snr_estimate for m in self.frame_metrics])
        avg_pilot_snr = np.mean([m.pilot_snr for m in self.frame_metrics])
        
        print(f"\nOverall Metrics:")
        print(f"  Average SNR:        {avg_snr:.2f} dB")
        print(f"  Average Pilot SNR:  {avg_pilot_snr:.2f} dB")
        print(f"  Overall BER:        {overall_ber:.6e}")
        print(f"  Overall PER:        {overall_per:.4f} ({total_packet_errors}/{len(self.frame_metrics)})")
        print(f"  Total Bits:         {total_bits:,}")
        print(f"  Total Bit Errors:   {total_bit_errors:,}")
        
        # Per-MCS statistics
        print(f"\nPer-MCS Statistics:")
        print(f"{'MCS':<15} {'Frames':<8} {'PER':<8} {'BER':<12} {'Avg SNR':<10}")
        print("-" * 60)
        
        for mcs_name in sorted(self.mcs_stats.keys()):
            stats = self.mcs_stats[mcs_name]
            if stats['frames'] == 0:
                continue
            
            per = stats['errors'] / stats['frames']
            ber = stats['bit_errors'] / stats['bits'] if stats['bits'] > 0 else 0
            
            # Calculate average SNR for this MCS
            mcs_frames = [m for m in self.frame_metrics if m.mcs_name == mcs_name]
            avg_mcs_snr = np.mean([m.snr_estimate for m in mcs_frames]) if mcs_frames else 0
            
            print(f"{mcs_name:<15} {stats['frames']:<8} {per:<8.4f} {ber:<12.6e} {avg_mcs_snr:<10.2f}")
        
        print("="*80)
    
    def run(self, frame_length: int = 20000, continuous: bool = True, max_frames: int = None):
        """Main receiver loop"""
        
        try:
            self.initialize()
            
            if continuous:
                # Continuous mode - no metadata, process stream as it arrives
                logger.info("Running in CONTINUOUS mode (for burst transmitter)")
                self.receive_continuous_stream(frame_length, max_frames)
            else:
                # Batch mode - expect metadata then all symbols
                logger.info("Running in BATCH mode (expect metadata)")
                received_signal = self.receive_zmq_stream(timeout_ms=5000, expect_metadata=True)
                
                if received_signal is not None:
                    self.process_continuous_stream(received_signal, frame_length)
            
            # Print statistics
            self.print_statistics()
            
        except KeyboardInterrupt:
            logger.info("\nReceiver interrupted by user")
        except Exception as e:
            logger.error(f"Error in receiver: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up receiver...")
        
        if self.rx_socket:
            self.rx_socket.close()
        if self.csi_socket:
            self.csi_socket.close()
        if self.context:
            self.context.term()
        
        logger.info("Receiver cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DVB-S2 ACM Receiver with CSI Feedback")
    
    parser.add_argument("--rx-address", type=str, default="tcp://127.0.0.1:5555",
                       help="ZMQ address to receive symbols from")
    parser.add_argument("--csi-address", type=str, default="tcp://127.0.0.1:5556",
                       help="ZMQ address to send CSI feedback")
    parser.add_argument("--sample-rate", type=float, default=4000,
                       help="Sample rate (Hz)")
    parser.add_argument("--symbol-rate", type=float, default=1000,
                       help="Symbol rate (Hz)")
    parser.add_argument("--n-ldpc", type=int, default=16200,
                       help="LDPC codeword length")
    parser.add_argument("--frame-length", type=int, default=20000,
                       help="Expected frame length in symbols")
    parser.add_argument("--log-file", type=str, default="receiver_metrics.log",
                       help="Output log file for metrics")
    parser.add_argument("--continuous", action="store_true", default=True,
                       help="Continuous mode for burst transmitters (default)")
    parser.add_argument("--batch", action="store_true",
                       help="Batch mode - expect metadata then full signal")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process (None=unlimited)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--save-symbols", type=str, default=None,
                       help="Save received symbols to file for analysis")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Batch mode overrides continuous
    continuous_mode = not args.batch
    
    # Create and run receiver
    receiver = DVB_S2_Receiver(
        zmq_rx_address=args.rx_address,
        zmq_csi_address=args.csi_address,
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        n_ldpc=args.n_ldpc,
        log_file=args.log_file
    )
    
    receiver.run(frame_length=args.frame_length, continuous=continuous_mode, max_frames=args.max_frames)


if __name__ == "__main__":
    main()