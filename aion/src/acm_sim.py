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
# mcs_table = [
#     {'name':'BPSK-1/4',   'mod':'BPSK',  'code_rate':1/4, 'snr':0.5},
#     {'name':'QPSK-1/3',   'mod':'QPSK',  'code_rate':1/3, 'snr':1.5},
#     {'name':'8APSK-2/3',  'mod':'8APSK','code_rate':2/3, 'snr':5},
#     {'name':'16APSK-3/4', 'mod':'16APSK','code_rate':3/4,'snr':8},
#     {'name':'32APSK-9/10','mod':'32APSK','code_rate':9/10,'snr':12},
# ]

mcs_table = [
    # QPSK
    {'name':'QPSK-1/4',   'mod':'QPSK',  'code_rate':1/4, 'snr':0.5},
    {'name':'QPSK-1/3',   'mod':'QPSK',  'code_rate':1/3, 'snr':0.8},
    {'name':'QPSK-2/5',   'mod':'QPSK',  'code_rate':2/5, 'snr':1.1},
    {'name':'QPSK-1/2',   'mod':'QPSK',  'code_rate':1/2, 'snr':1.5},
    {'name':'QPSK-3/5',   'mod':'QPSK',  'code_rate':3/5, 'snr':1.8},
    {'name':'QPSK-2/3',   'mod':'QPSK',  'code_rate':2/3, 'snr':2.1},
    {'name':'QPSK-3/4',   'mod':'QPSK',  'code_rate':3/4, 'snr':2.5},
    {'name':'QPSK-4/5',   'mod':'QPSK',  'code_rate':4/5, 'snr':2.8},
    {'name':'QPSK-5/6',   'mod':'QPSK',  'code_rate':5/6, 'snr':3.0},
    {'name':'QPSK-8/9',   'mod':'QPSK',  'code_rate':8/9, 'snr':3.5},
    # 8APSK
    {'name':'8APSK-3/5',  'mod':'8APSK','code_rate':3/5, 'snr':4.5},
    {'name':'8APSK-2/3',  'mod':'8APSK','code_rate':2/3, 'snr':5.0},
    {'name':'8APSK-3/4',  'mod':'8APSK','code_rate':3/4, 'snr':5.5},
    {'name':'8APSK-5/6',  'mod':'8APSK','code_rate':5/6, 'snr':6.0},
    {'name':'8APSK-8/9',  'mod':'8APSK','code_rate':8/9, 'snr':6.5},
    # 16APSK
    {'name':'16APSK-2/3', 'mod':'16APSK','code_rate':2/3,'snr':7.5},
    {'name':'16APSK-3/4', 'mod':'16APSK','code_rate':3/4,'snr':8.0},
    {'name':'16APSK-4/5', 'mod':'16APSK','code_rate':4/5,'snr':8.5},
    {'name':'16APSK-5/6', 'mod':'16APSK','code_rate':5/6,'snr':9.0},
    {'name':'16APSK-8/9', 'mod':'16APSK','code_rate':8/9,'snr':9.5},
    # 32APSK
    {'name':'32APSK-3/4','mod':'32APSK','code_rate':3/4,'snr':11.0},
    {'name':'32APSK-4/5','mod':'32APSK','code_rate':4/5,'snr':11.5},
    {'name':'32APSK-5/6','mod':'32APSK','code_rate':5/6,'snr':12.0},
    {'name':'32APSK-8/9','mod':'32APSK','code_rate':8/9,'snr':12.5},
]


# MODCOD_TABLE = [
#     # (mod, rate, index)
#     ("QPSK", "1/4", 1),
#     ("QPSK", "1/3", 2),
#     ("QPSK", "2/5", 3),
#     ("QPSK", "1/2", 4),
#     ("QPSK", "3/5", 5),
#     ("QPSK", "2/3", 6),
#     ("QPSK", "3/4", 7),
#     ("QPSK", "4/5", 8),
#     ("QPSK", "5/6", 9),
#     ("QPSK", "8/9", 10),
#     ("8PSK", "3/5", 12),
#     ("8PSK", "2/3", 13),
#     ("8PSK", "3/4", 14),
#     ("8PSK", "5/6", 15),
#     ("8PSK", "8/9", 16),
#     ("16APSK", "2/3", 18),
#     ("16APSK", "3/4", 19),
#     ("16APSK", "4/5", 20),
#     ("16APSK", "5/6", 21),
#     ("16APSK", "8/9", 22),
#     ("32APSK", "3/4", 24),
#     ("32APSK", "4/5", 25),
#     ("32APSK", "5/6", 26),
#     ("32APSK", "8/9", 27),
# ]

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
