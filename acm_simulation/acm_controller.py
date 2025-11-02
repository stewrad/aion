"""
ACM Simulation - Adaptive Coding and Modulation Controller
Selects modulation and coding scheme based on channel conditions (SNR/CSI)
"""

from enum import IntEnum
from gnuradio import digital
import numpy as np


class ModulationScheme(IntEnum):
    """Available modulation schemes"""
    BPSK = 1    # 1 bit/symbol
    QPSK = 2    # 2 bits/symbol
    PSK8 = 3    # 3 bits/symbol
    QAM16 = 4   # 4 bits/symbol
    QAM64 = 6   # 6 bits/symbol


class CodingScheme(IntEnum):
    """Available coding rates (x/y means x/y rate)"""
    RATE_1_2 = 0   # Rate 1/2 - most robust
    RATE_2_3 = 1   # Rate 2/3
    RATE_3_4 = 2   # Rate 3/4
    RATE_5_6 = 3   # Rate 5/6 - least robust, highest throughput


class ACMController:
    """
    Adaptive Coding and Modulation Controller.
    Maps SNR to optimal modulation and coding scheme (MCS).
    """
    
    # ACM lookup table: SNR thresholds for each MCS
    # Format: (min_snr_db, modulation, coding_rate, spectral_efficiency)
    ACM_TABLE = [
        # SNR (dB), Modulation, Coding, Spectral Efficiency (bits/s/Hz)
        (-2.0,  ModulationScheme.BPSK,  CodingScheme.RATE_1_2, 0.50),
        (1.0,   ModulationScheme.QPSK,  CodingScheme.RATE_1_2, 1.00),
        (4.0,   ModulationScheme.QPSK,  CodingScheme.RATE_2_3, 1.33),
        (6.5,   ModulationScheme.QPSK,  CodingScheme.RATE_3_4, 1.50),
        (8.5,   ModulationScheme.QPSK,  CodingScheme.RATE_5_6, 1.67),
        (11.0,  ModulationScheme.PSK8,  CodingScheme.RATE_2_3, 2.00),
        (13.0,  ModulationScheme.PSK8,  CodingScheme.RATE_3_4, 2.25),
        (15.0,  ModulationScheme.QAM16, CodingScheme.RATE_2_3, 2.67),
        (17.0,  ModulationScheme.QAM16, CodingScheme.RATE_3_4, 3.00),
        (19.0,  ModulationScheme.QAM16, CodingScheme.RATE_5_6, 3.33),
        (22.0,  ModulationScheme.QAM64, CodingScheme.RATE_3_4, 4.50),
        (25.0,  ModulationScheme.QAM64, CodingScheme.RATE_5_6, 5.00),
    ]
    
    def __init__(self, hysteresis_db=1.0):
        """
        Initialize ACM controller.
        
        Args:
            hysteresis_db: Hysteresis margin in dB to prevent oscillation
        """
        self.hysteresis_db = hysteresis_db
        self.current_mcs_index = 0
        self.snr_history = []
        self.max_history = 10
        
        print(f"[ACMController] Initialized with {len(self.ACM_TABLE)} MCS modes")
        self._print_acm_table()
    
    def _print_acm_table(self):
        """Print the ACM lookup table"""
        print("\\nACM Lookup Table:")
        print("-" * 70)
        print(f"{'SNR (dB)':<10} {'Modulation':<12} {'Code Rate':<12} {'Spectral Eff':<12}")
        print("-" * 70)
        for snr, mod, code, eff in self.ACM_TABLE:
            mod_name = mod.name
            code_name = code.name.replace('RATE_', '')
            print(f"{snr:<10.1f} {mod_name:<12} {code_name:<12} {eff:<12.2f}")
        print("-" * 70)
    
    def select_mcs(self, snr_db):
        """
        Select MCS based on current SNR estimate.
        Uses hysteresis to prevent rapid switching.
        
        Args:
            snr_db: Current SNR estimate in dB
            
        Returns:
            Tuple of (mcs_index, modulation, coding_rate, spectral_efficiency)
        """
        # Update SNR history
        self.snr_history.append(snr_db)
        if len(self.snr_history) > self.max_history:
            self.snr_history.pop(0)
        
        # Use averaged SNR for stability
        avg_snr = np.mean(self.snr_history)
        
        # Apply hysteresis
        if self.current_mcs_index < len(self.ACM_TABLE) - 1:
            # Check if we should upgrade to higher MCS
            next_mcs = self.ACM_TABLE[self.current_mcs_index + 1]
            if avg_snr >= next_mcs[0] + self.hysteresis_db:
                self.current_mcs_index += 1
        
        if self.current_mcs_index > 0:
            # Check if we should downgrade to lower MCS
            current_mcs = self.ACM_TABLE[self.current_mcs_index]
            if avg_snr < current_mcs[0] - self.hysteresis_db:
                self.current_mcs_index -= 1
        
        # Return current MCS
        mcs = self.ACM_TABLE[self.current_mcs_index]
        return (self.current_mcs_index, mcs[1], mcs[2], mcs[3])
    
    def get_constellation(self, modulation):
        """
        Get GNU Radio constellation object for given modulation.
        
        Args:
            modulation: ModulationScheme enum value
            
        Returns:
            GNU Radio constellation object
        """
        if modulation == ModulationScheme.BPSK:
            return digital.constellation_bpsk()
        elif modulation == ModulationScheme.QPSK:
            return digital.constellation_qpsk()
        elif modulation == ModulationScheme.PSK8:
            return digital.constellation_8psk()
        elif modulation == ModulationScheme.QAM16:
            return digital.constellation_16qam()
        elif modulation == ModulationScheme.QAM64:
            # GNU Radio 3.8+ has 64QAM, fallback to custom if not available
            try:
                return digital.constellation_64qam()
            except AttributeError:
                # Create custom 64-QAM constellation
                amps = [-7, -5, -3, -1, 1, 3, 5, 7]
                constellation_points = []
                for a in amps:
                    for b in amps:
                        constellation_points.append(complex(a, b))
                return digital.constellation_calcdist(
                    constellation_points, [], 4, 1
                ).base()
        else:
            return digital.constellation_qpsk()  # Default fallback
    
    def get_fec_object(self, coding_rate, frame_size=1024):
        """
        Get GNU Radio FEC encoder/decoder objects for given coding rate.
        Uses convolutional codes for this implementation.
        
        Args:
            coding_rate: CodingScheme enum value
            frame_size: FEC frame size in bits
            
        Returns:
            Tuple of (encoder_object, decoder_object)
        """
        from gnuradio import fec
        
        # Convolutional code polynomials
        k = 7
        rate = 2
        polys = [79, 109]  # Octal, rate 1/2
        
        # Create encoder
        encoder = fec.cc_encoder_make(
            frame_size=frame_size,
            k=k,
            rate=rate,
            polys=polys
        )
        
        # Create decoder (Viterbi)
        decoder = fec.cc_decoder.make(
            frame_size=frame_size,
            k=k,
            rate=rate,
            polys=polys,
            mode=fec.CC_STREAMING,
            # padding=False
            padded=False
        )
        
        return encoder, decoder
    
    def calculate_throughput(self, symbol_rate, spectral_efficiency):
        """
        Calculate throughput for given parameters.
        
        Args:
            symbol_rate: Symbol rate in symbols/second
            spectral_efficiency: Spectral efficiency in bits/s/Hz
            
        Returns:
            Throughput in bits/second
        """
        return symbol_rate * spectral_efficiency
    
    def get_current_mcs_info(self):
        """Get information about current MCS"""
        mcs = self.ACM_TABLE[self.current_mcs_index]
        return {
            'mcs_index': self.current_mcs_index,
            'min_snr_db': mcs[0],
            'modulation': mcs[1].name,
            'coding_rate': mcs[2].name,
            'spectral_efficiency': mcs[3],
            'bits_per_symbol': int(mcs[1])
        }
