class RXFlowgraph(gr.top_block):
    def __init__(self, mcs):
        gr.top_block.__init__(self)
        demod = {
            'BPSK': digital.psk.psk_demod(2),
            'QPSK': digital.psk.psk_demod(4),
            '16QAM': digital.qam.qam_demod(16)
        }[mcs['mod']]
        
        dec = fec.decoder(encoder_obj_list=fec.cc_decoder_make(7, 2, 3))
        
        self.src = blocks.vector_source_c([])
        self.demod = demod
        self.dec = dec
        self.snk = blocks.vector_sink_b()
        self.connect(self.src, self.demod, self.dec, self.snk)
