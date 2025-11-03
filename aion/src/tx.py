from gnuradio import gr, blocks, digital, fec

class TXFlowgraph(gr.top_block):
    def __init__(self, mcs):
        gr.top_block.__init__(self)
        mod = {
            'BPSK': digital.psk.psk_mod(2),
            'QPSK': digital.psk.psk_mod(4),
            '16QAM': digital.qam.qam_mod(16)
        }[mcs['mod']]
        
        # Use FEC Encoder
        enc = fec.encoder(encoder_obj_list=fec.cc_encoder_make(7, 2, 3))
        
        self.src = blocks.vector_source_b([])
        self.enc = fec.encoder(enc)
        self.mod = mod
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, 32000)
        self.snk = blocks.vector_sink_c()
        
        self.connect(self.src, self.enc, self.mod, self.throttle, self.snk)
