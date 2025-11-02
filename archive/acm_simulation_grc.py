#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ACM Simulation
# Author: ACM Research
# Description: Adaptive Coding and Modulation Simulation
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
from gnuradio import blocks, gr
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import acm_simulation_grc_packet_gen as packet_gen  # embedded python block
import numpy
import sip
import threading


def snipfcn_snippet_path(self):
    sys.path.insert(0, '/home/arsewar/Documents/ee448')


def snippets_main_after_init(tb):
    snipfcn_snippet_path(tb)

class acm_simulation_grc(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "ACM Simulation", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("ACM Simulation")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "acm_simulation_grc")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 10
        self.samp_rate = samp_rate = 1000000
        self.noise_voltage = noise_voltage = 1.0/numpy.sqrt(2.0*10**(snr_db/10.0))

        ##################################################
        # Blocks
        ##################################################

        self._snr_db_range = qtgui.Range(-5, 30, 0.5, 10, 200)
        self._snr_db_win = qtgui.RangeWidget(self._snr_db_range, self.set_snr_db, "SNR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._snr_db_win)
        self.qtgui_const_sink = qtgui.const_sink_c(
            2048, #size
            "Constellation", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink.set_update_time(0.10)
        self.qtgui_const_sink.set_y_axis((-2), 2)
        self.qtgui_const_sink.set_x_axis((-2), 2)
        self.qtgui_const_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink.enable_autoscale(False)
        self.qtgui_const_sink.enable_grid(True)
        self.qtgui_const_sink.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink.set_line_label(i, labels[i])
            self.qtgui_const_sink.set_line_width(i, widths[i])
            self.qtgui_const_sink.set_line_color(i, colors[i])
            self.qtgui_const_sink.set_line_style(i, styles[i])
            self.qtgui_const_sink.set_line_marker(i, markers[i])
            self.qtgui_const_sink.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_win = sip.wrapinstance(self.qtgui_const_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_win)
        self.packet_gen = packet_gen.blk(packet_rate=100)
        self.digital_constellation_decoder = digital.constellation_decoder_cb(digital.constellation_qpsk())
        self.digital_chunks_to_symbols = digital.chunks_to_symbols_bc([-1-1j, -1+1j, 1-1j, 1+1j], 1)
        self.channels_channel_model = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0,
            block_tags=False)
        self.blocks_throttle = blocks.throttle(gr.sizeof_char*1, samp_rate,True)
        self.blocks_message_debug = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_file_sink = blocks.file_sink(gr.sizeof_char*1, '/tmp/acm_output.bin', False)
        self.blocks_file_sink.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.packet_gen, 'packet_info'), (self.blocks_message_debug, 'print'))
        self.connect((self.blocks_throttle, 0), (self.digital_chunks_to_symbols, 0))
        self.connect((self.channels_channel_model, 0), (self.digital_constellation_decoder, 0))
        self.connect((self.channels_channel_model, 0), (self.qtgui_const_sink, 0))
        self.connect((self.digital_chunks_to_symbols, 0), (self.channels_channel_model, 0))
        self.connect((self.digital_constellation_decoder, 0), (self.blocks_file_sink, 0))
        self.connect((self.packet_gen, 0), (self.blocks_throttle, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "acm_simulation_grc")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.set_noise_voltage(1.0/numpy.sqrt(2.0*10**(self.snr_db/10.0)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle.set_sample_rate(self.samp_rate)

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage
        self.channels_channel_model.set_noise_voltage(self.noise_voltage)




def main(top_block_cls=acm_simulation_grc, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    snippets_main_after_init(tb)
    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
