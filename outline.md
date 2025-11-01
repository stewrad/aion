# To Do:

## Overview
1. Build an ACM system similar to DVBS-2
    - pilot symbols (pi/2-bpsk)
    - modcod 
    - data frames to transmit ?? 
    - pilot symbols throughout? 
2. Code the FEC for Rx decoding
    - Mimick DVBS-2? LDPC at varying coding rates ? 
3. Code Channel simulation models in Python
    - Lectures
    - Literature review
4. Connect/Configure USRP N210 for Tx/Rx 
5. Research reinforcement learning, identify a way to improve ACM with this 
    - literature review
    - potentially code this in Python 

Big unanswered questions: 
1. How will I feed back the channel state information to my transmitter? 
    - control channel 
    - feedback from pilot symbols interspersed throughout 
2. What data will I transmit ?
3. Reinforcement Learning - How? 
4. What sort of link am I going to simulate? Uplink/Downlink or another communication type (4G LTE/5G NR)? 
    a. What frequency range / band will I be operating in? 



## System Architecture (logical modules)
### 1. Tx PHY (GNU Radio / Python)
- Packet Generator
- FEC encoder: likely LDPC
- Modulator: BPSK/QPSK/16QAM with RRC pulse-shaping (configurable roll-off)
- Pilot insertion (for channel estimate) + preamble for coarse sync (PI/2-BPSK)
- UHD sink to N210 Tx
### 2. Rx PHY (GNU Radio / Python)
- UHD source from N210 Rx
- Front-end: DC removal, AGC, BPF
- Packet detect (preamble correlation), frame extraction
- Pilot-based channel estimation -> SNR estimator
- Soft demodulator (LLR generation) for the FEC decoder
- FEC decoder + CRC check
- KPI measurements 
### 3. ACM controller (Python)
- Map measured SNR (and optionally history/variance) to MCS (modulation + code rate)
- Can be a simple thresholds table or an ML model
- Signals Tx to changes MCS between frames 
### 4. Control & Logging (ZeroMQ or UDP RPC)
- Low-latency messaging bus for SNR reports, ACKs, MCS updates, and metrics logging
- CSV/JSON log files with timestamps, SNR, PER, throughput 

## Frame & Packet Format (recommended) 
- **Preamble**: repeated known sequence for coarse timing and frequency offset estimatino (e.g. 128 $\mu s$ of Zadoff-Chu or Barker-like pattern)
- **Pilot block**: single OFDM-like pilot or pilot symbols every N symbols for channel estimatino (if single-carrier, put pilot symbols in-line)
- **Header**: MCS indicator (8 bits), packet ID (16 bits), payload length (16 bits), redundancy version (if IR)(3 bits), CRC-16 of header
- **Payload**: FEC-encoded bits (rate depends on MCS)
- **CRC**: CRC-32 or CRC-16 for payload integrity check (receiver uses for ACK/NACK decision)

## FEC recommended parameters
- **Simple start**: convolution code (1/2 rate, constraint lenght 7) + Viterbi decoder; easy to implement in GNU Radio. Use puncturing to get rates {1/2, 2/3, 3/4}
- **Research-grade**: rate-compatible LDPC if you want true IR. Use GNU Radio LDPC blocks

## Overlaying "real RF" (replay / recorded interference)
You want to evaluate performance with realistic interference. Options inside closed-loop 
1. Recorded capture replay: record an RF capture (IQ) of a real environment (Wi-Fi handshake, LTE noise, or other). Re-play the IQ file via a separate USRP/host into the combiner that feeds the Rx. If you have only one N210, you can save the IQ file and stream it into the combiner using a small external signal generator or the same N210 by time-multiplexing (more complex)
2. Synthetic interferer: Use a GNU Radio flowgraph to generate a Wi-Fi-like OFDM frames / narrowband interferers and feed them into the combiner with adjustable power 
3. Loopback mixing: Split your Tx into two branches: one branch is the desired signal; the other branch is run through an impairment block (delay, multipath, additional noise) and recombined to simulate multi-user interfrence 

## Implementation Roadmap (concrete steps)
### 1. Environment & sanity checks
- Install UHD, GNU Radio, and confirm uhd_find_devices and uhd_usrp_probe
- Create a simple GNU Radio transmit -> coax -> receive loop (cable with with 60 dB attenuation) that transmits a single tone and receives it 
### 2. Simple packet Tx/Rx
- Implement a Tx flowgraph: preamble + BPSK payload + CRC -> RRC -> UHD sink
- Implement a Rx flowgraph: UHD source -> matched filter -> preamble corr -> symbol sync -> hard demod -> CRC check
- Verify correct packet reception when attenuation is low 
### 3. Add pilot-based channel estimation and SNR reporting
- Insert pilots and compute per-frame SNR estimate at Rx (e.g. using pilot symbol power vs noise floor) 
- Stream SNR reports via ZeroMQ to the controller module. Log values. 
### 4. Add FEC (convolutional) & soft demod
- Replace hard demod with soft demod (LLR) flow to feed a Viterbi decoder
- Verify FEC reduces BER as expected in AWGN simulation (also test with loopback)
### 5. Implement HARQ: (Optional)
- Tx retains each sent codeword in a buffer
- When Rx NACKs, Tx retransmits same codeword; Rx soft-combines LLRs (sum) and attempts decode again 
- Implement retransmission counters and stop after max retransmissions
### 6. Implement ACk/NACK PHYS (Optional)
- Create a robust ACK/NACK fraem type and small Tx burst from Rx host back to the Tx host (via combiner loopback)
- Verify reliability of feedback (log ACK/NACK success rates). 
### 7. Implement ACM 
- Start with an SNR threshold table mapping to {BPSK_R1/2, QPSK_R1/2, 16QAM_R3/4}
- Use controller to select the MCS for each new packet and inform Tx via ZeroMQ
- Test link under steady SNR to ensure adaptation switches MCS appropriately 
### 8. Overlay replayed/recorded RF (Optional)
- Prepare an IQ capture file (real or synthetic). Use a GNU Radio flowgraph to stream this IQ into the combiner and into Rx chain along with Tx signal. 
- Sweep the interference power and observe PER and throughput 
### 9. Measurement automation 
- Write scripts to sweep attenuation SNR steps, interference power, and record PER/BER, throughput, avg retransmissions, latency histograms
- Save CSV logs and produce plots 
### 10. Optional Add IR HARQ & LDPC
- Impelment IR using a mother LDPC code and different redundancy versions. Use an LDPC library or GNU Radio LDPC blocks. This increases complexity but shows modern HARQ performance gains 
### 11. Implement Reinforcement Learning 
- ...

## Experiments and What to Measure
- BER/PER vs SNR curves for each MCS
- Throughput (goodput) vs SNR comparing fixed MCS, ACM-only, and ACM+RL 
- Robustness versus interference: fix SNR, vary interfer power, measure PER and throughput
- Compare ACM vs ACM+RL 
    - Measure RL-based MCS selection **increases long-term goodput** vs baselines in stationary and nonstationary channels
    - Measure sample efficiency (how fast RL converges) and stability (oscillations, safety)
    - Measure RL robustness to realistic interference and channel nonstationarity 
    - Understand reward design, state features, and exploration strategy impacts via ablation  

## Evaluation Metrics and Expected Observations
- ACM improves average throughput by switching to higher order MCS at higher SNRs; combined with RL you get reliable higher throughput across a wider SNR range. 

## ACM MCS Code Example:
```python
ACM_TABLE = [
    {"mcs":"BPSK_R12", "min_snr_db": -5},
    {"mcs":"QPSK_R12", "min_snr_db": 5},
    {"mcs":"QPSK_R34", "min_snr_db": 8},
    {"mcs":"16QAM_R34","min_snr_db": 14},
]
def choose_mcs(snr_db):
    # choose highest MCS where snr >= min_snr
    for entry in reversed(ACM_TABLE):
        if snr_db >= entry["min_snr_db"]:
            return entry["mcs"]
    return ACM_TABLE[0]["mcs"]
```

## Safety and Compliance (**IMPORTANT**)
- **DO NOT** connect antennas or radiate during these tests unless you are legally authorized. Use coax, attenuators, and shielded enclosure
- Keep attenuation high when summing signals into the Rx port to avoid damaging the Rx front end

## Deliverables
- a working GNU Radio Tx and Rx flowgraph in closed-loop over coax
- a Python controller that performs ACM and logs metrics
- scripts that reproduce BER/PER vs SNR and interference plots
- Demonstrated overlay of a recorded real-RF interferer with variable power and measured impact on link performance
- Reinforcement learning implementation in Python 



## Literature Review: 
*On realization of reliable link layer protocols with guaranteed bounds using SDR* S. Soltani et al. (2011)
- Experimental platform with USRP front-end to evaluate reliable link-layer schemes in real SDR environment

*LTE PHY Layer Vulnerability Analysis and Testing Using Open-Source SDR Tools* R. M. Rao, S. Ha, V. Marojevic, J. H. Reed (2017)
- how to build SDR-based testing infrastructure for PHY protocols including interference injection, subsystem testing

*Machine Learning for Physical Layer in 5G and beyond* 
- Survey on ML-based soultison for PHY-layer challenges (channel estimation, mod classification, link adapation, etc.)

*Adaptive Modulation and Coding based on Reinforcement Learning for 5G Networks* (2019, Mota et al.)
- Q-learning based RL framework for mapping CQI -> MCS to maximize spectral efficiency while maintaining low BLER

*Deep Reinforcement Learning‑Based Modulation and Coding Scheme Selection in Cognitive Heterogeneous Networks* (2019, Zhang et al.)
- Deep RL approach for MCS selection in cognitive heterogeneous networks 

*Reinforcement Learning for Delay Sensitive Uplink Outer‑Loop Link Adaptation* (2022, Nokia)
- RL applied to outer-loop link adapation (OLLA) in pulink 

*Deep Reinforcement Learning‑Based Adaptive Modulation for OFDM Underwater Acoustic Communication System* (2023, EURASIP)
- DRL for adapative modulation based on time-varying channels 

*Reinforcement Learning for Link Adaptation and Channel Selection in LEO Satellite Cognitive Communications* (2023) 
- RL applied for channel & MCS selection in LEO satellite/cognitive scenario 

*Reinforcement Learning for Efficient and Tuning Free Link Adaptation* (2020, Saxena et al.)
- Latent Thompson Sampling RL scheme for MCS/link adaptation, no heavy tuning 

*Joint Adaptive Transmission and Numerology Selection for 5G NR PDSCH with DQN‑based Reinforcement Learning Solution* (2024)
- RL (DQN) for joint MCS + numerology selection in 5G NR PDSCH

