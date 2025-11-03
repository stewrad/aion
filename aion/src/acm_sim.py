"""
Simple DVB-S2-style ACM simulation using GNU Radio concepts in Python.
"""
import numpy as np
from gnuradio import gr, blocks, digital, fec, channels

# User-defined packages 
from .packet_generator import generate_packet

# -----------------------
# MCS Table
# -----------------------
mcs_table = [
    {'name':'BPSK-1/4',   'mod':'BPSK',  'code_rate':1/4, 'snr_threshold':0.5},
    {'name':'QPSK-1/3',   'mod':'QPSK',  'code_rate':1/3, 'snr_threshold':1.5},
    {'name':'8APSK-2/3',  'mod':'8APSK','code_rate':2/3, 'snr_threshold':5},
    {'name':'16APSK-3/4', 'mod':'16APSK','code_rate':3/4,'snr_threshold':8},
    {'name':'32APSK-9/10','mod':'32APSK','code_rate':9/10,'snr_threshold':12},
]


# Primary orchestrator loop 
snr_values = np.linspace(0, 20, 10)
mcs_table = 'BPSK'
for snr in snr_values:
    current_mcs = mcs_table[0]
    tx = TXFlowgraph(current_mcs)
    tx.start(); tx.wait()
    tx_out = np.array(tx.snk.data())
    
    rx_in = awgn(tx_out, snr)
    
    rx = RXFlowgraph(current_mcs)
    rx.start(); rx.wait()
    rx_out = np.array(rx.snk.data())
    
    ber = np.mean(rx_out != tx.src.data())
    new_mcs = acm_controller(snr, current_mcs, mcs_table)
