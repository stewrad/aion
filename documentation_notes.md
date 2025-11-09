1. Packet Generator
Input: Packet size
Output: Packets in a bit array 

2. Segmentation and Framing
Sequence number, frame sync?, MCS? 

bits_payload,header,crc
b_total = bits_payload,header,crc (total information)

K = LDPC information bits per base codeword (before shortening/puncturing)
N = LDPC coded bits ouput per codeword (N = K / rate)
r = K / N (LDPC code rate)

m = bits per modulation symbol (e.g. QPSK=2)
L_sym = sampels per symbol (oversampling), not needed for segmentation but relevant for Output
num_cw = number of LDPC codewords per PHY frame

LDPC either supports fixed K or shortening/puncturing to achieve different rates/lengths 

DVBS-2 LDPC 

Block type | info bits (K) | Codeword bits (N) 
Normal     | k = 64,800    | N = 64,800 / rate
Short      | k = 16,200    | N = 16,200 / rate 

How to Segment: 
1. Choose frame type (normal or short)
2. Pick an MCS 
3. for that rate, use corresponding K (information bits per LDPC codeword)
4. Segment your payload into chunks of K bits 
5. If the last chunk < K bits, use shortening (fill missing bits with zeros before LDPC)


Base LDPC frame sizes:
  - Normal frame: N = 64800 bits
  - Short frame:  N = 16200 bits

Supported code rates (and corresponding K):
  Rate: 1/4  → K = 16200 (short) / 40500 (normal)
  Rate: 1/3  → K = 21600 / 5400
  Rate: 2/5  → K = 25920 / 6480
  Rate: 1/2  → K = 32400 / 8100
  Rate: 3/5  → K = 38880 / 9720
  Rate: 2/3  → K = 43200 / 10800
  Rate: 3/4  → K = 48600 / 12150
  Rate: 4/5  → K = 51840 / 12960
  Rate: 5/6  → K = 54000 / 13500
  Rate: 8/9  → K = 57600 / 14400
  Rate: 9/10 → K = 58320 / 14580

## IMPORTANT TO NOTE: 
- I decided to utilize short frame (N=16200) to reduce computational load on my simulation. This means I can no longer use 9/10 code rate. See full mcs_table below: 

## LDPC Framing Steps:
#### 1. Collection application/transport packets -> output bit array
#### 2. Create PHY Header
- SoF from SRG LFSR
    - unscrambled
- Pilots from SRG LFSR
    - SRG LFSR scrambler
- MCS Indices as Walsh codes
    - SRG LFSR scrambler 
#### 3. Segment for LDPC: 
- Choose the code rate (r) and K based on MCS
- segment PHY frame into chunks of K bits
- IF final segment < K, pad with 0s
#### 4. LDPC encode each segment
- each segment -> encoded to N bits
#### 5. Combine encoded frames (e.g. time multiplex several per burst)
- Optionally bit interleave
#### 6. Map bits to modulation symbols
#### 7. Insert modulated pilots/MCS index/SoF symbols 
- pilots every 16 symbols
    - pilot density can be determined by MCS as well, e.g. 1p per 32d for QPSK or 1p per 16d for 16APSK
- Optionally apply PLS here 
#### 8. Apply RRC filter, upsample, apply DC offset 
#### 9. Transmit on SDR, here F_c is applied by SDR


Integration with GNU Radio FEC Encoding:
- GNU Radio has fec.ldpc_encoder/decoder blocks
- you must segment into groups of K bits before feeding to GNU Radio

Example segmenation in Python 
```python
def segment_bits_for_ldpc(bits, K):
    segments = []
    while len(bits) > K:
        segments.append(bits[:K])
        bits = bits[K:]
    if len(bits) > 0:
        # pad or shorten last segment
        pad_len = K - len(bits)
        bits += [0] * pad_len
        segments.append(bits)
    return segments
```

Note: 
Implement a ACM controller block that 
- Reads channel SNR feedback
- selects appropriate MCS
- updates FEC encoder and modulation mapper configuration dynamically 


## Start of Frame and Pilot Symbol Pseudorandom Sequences:
1. Linear Feedback Shift Register (LFSR) 
    - generate a maximal-length pseudorandom sequence (m-sequence) 
2. Gold or Kasami Sequences
    - combines two LFSRs -> longer sequences, better cross-correlation 
    - Useful when you need multiple sequences (e.g. forward + return link pilots)
3. Simple PRBS
    - use NumPy to make a pseudorandom sequence
```python
import numpy as np

np.random.seed(42)  # fixed seed for reproducibility
sof_bits = np.random.randint(0, 2, size=128)  # 128-bit SoF
pilot_bits = np.random.randint(0, 2, size=16) # 16-bit pilot
```

Interleaving pilot symbols:
```python
pilot_interval = 16  # pilot every 16 data symbols
pilot_idx = 0
tx_symbols = []

for i, sym in enumerate(data_symbols):
    tx_symbols.append(sym)
    if (i+1) % pilot_interval == 0:
        tx_symbols.append(pilot_symbols[pilot_idx])
        pilot_idx = (pilot_idx + 1) % len(pilot_symbols)  # wrap around if needed

tx_symbols = np.array(tx_symbols)
```
-`pilot_interval` can be modulation-based 

Interspersing SoF:
```python
phy_frame_symbols = np.concatenate([sof_symbols, header_symbols, tx_symbols])
```

## Encoding MCS Indices: 
1. Direct binary encoding + CRC
    - Map MCS index to binary, e.g. 5 MCS levels -> 3 bits
    - optionally append 1-2 bit CRC
    - modulate with robust constellation (pi/2)
2. Orthogonal Sequences / Walsh Codes
    - map each MCS index to a short orthogonal sequence (like a Walsh or Hadamard code)
    - properties: 
        - good cross-correlation -> easy to detect in noisy channel
        - robust even at very low SNR
    - receiver performs correlation with all possible sequences -> selects highest correlation -> identifies MCS 

**Example**
| MCS | 4-bit Walsh Code |
|--------|--------------|
| 0 | 0001 | 
| 1 | 0010 | 
| 2 | 0100 | 
| 3 | 1000 | 
| 4 | 1111 | 

Receiver decoding:
1. Detect SoF -> frame aligned
2. Extract MCS symbols
3. If orthogonal/Walsh codes -> correlate with all possible codes, pick maximum\
4. Rx now knows **modulation order, LDPC code rate, pilot interval, etc.**

## Walsh Code Generation 
**Walsh codes** are derived from **Hadamard matrices**, which are square matrices with entries $\pm$1 and mutually orthogonal rows
- for **size N**, you get N orthogonal codes of length **N**
- N must be a power of 2: 2,4,8,16,...

a. Recursive generation 
Hadamard matrices can be generated recursively:

$$H_1 = [1], H_{2N} = \begin{bmatrix}H_N & H_N \\ H_N & -H_N\end{bmatrix}$$
- each row -> a Walsh code (after optional row permutations)
Python example:
```python
import numpy as np

def hadamard_matrix(n):
    """Recursive Hadamard matrix of size n (n must be power of 2)."""
    if n == 1:
        return np.array([[1]])
    else:
        H_n = hadamard_matrix(n//2)
        top = np.hstack([H_n, H_n])
        bottom = np.hstack([H_n, -H_n])
        return np.vstack([top, bottom])

# Generate Walsh codes for 8 sequences
N = 8
H = hadamard_matrix(N)
walsh_codes = H  # Each row is a Walsh code of length N
```
Scrambling: 
1. Generate a PRBS of same length N
```python
np.random.seed(42)
scrambler = np.random.choice([-1, 1], size=N)
```
2. Multiply Walsh row by scrambler (element-wise):
```python
walsh_scrambled = walsh_codes[3] * scrambler
```
Shift Register Generator (SRG):
- Designed using maximal-length polynomials (primitive polynomials over GF(2))
- produces maximal-lenght sequences (m-sequences) of lenght $2^n-1$ for n-stage registers
- Good autocorrelation properties: peak at zero lag, nearly zero elsewhere
- Near-flat spectral density -> better for pilots and SoF

Doing it with a SRG-style LFSR: 
1. Choose a primitive polynomial over GF(2)
    - Example: 
        - $x^7+x^6+1$
    - these define which register bits are **XORed** to form the feedback
    - Using primitive polynomial ensures maximal-length m-sequence: sequence length = $2^n-1$
```python
import numpy as np

def lfsr(seed, taps, n_bits):
    """
    n_bits: number of output bits to generate
    seed: initial state as list of 0/1, length = LFSR length
    taps: feedback tap positions (0-based)
    """
    sr = seed.copy()
    out = []
    for _ in range(n_bits):
        out.append(sr[-1])
        feedback = 0
        for t in taps:
            feedback ^= sr[t]
        sr = [feedback] + sr[:-1]
    return np.array(out)

# Example: 7-bit LFSR, taps at positions 6 and 5 (x^7 + x^6 + 1)
seed = [1,0,0,0,0,0,1]  # cannot be all zeros
taps = [6,5]  # zero-based indexing
sequence = lfsr(seed, taps, 127)  # maximal length 2^7-1 = 127
sequence = 2*sequence - 1  # map 0->-1, 1->1 for BPSK
```
## Why to Scramble:
1. Improve spectral flatness (spectral shaping) 
    - transmitting long sequences of zeros or ones and structured payloads can cause long runs of the same symbol or create spectral spikes, or DC bias, in the transmitted signal
    - scrambling evenly distributes power across frequencies, hence the flat spectrum 
2. Reduce repeated patterns 

How I'll apply scrambling:
1. SoF: 
    - keep unscrambled but generate from SRG LFSR
2. MCS Index: 
    - scramble with known short scrambler 
    - generated as Walsh codes 
3. Pilots: 
    - Generate with SRG LFSR, scramble with short scrambler 
    - Descramble before channel estimation 
4. Payload:
    - Scramble with SRG LFSR for spectral flattening and randomization 

## Physical Layer Framing (PLFRAME)
Each PLFRAME in DVB-S2 corresponds to 
```cpp
[ SoF | PLS | (optional pilots) | LDPC-encoded payload ]
```
- the LDPC payload always has a fixed length of K bits (before encoding)
- the encoded output (N bits) depends on the code rate chosen by the MCS 

Example MCS Table:
| Modulation | Code Rate | K (LDPC Input Bits) | N (encoded bits) |
| ---------- | --------- | ------------------- | ---------------- |
|QPSK | 1/2 | 32400 | 64800 | 
| 8PSK | 3/4 | 48600 | 64800 | 
| 16APSK | 2/3 | 43200 | 64800 | 
| 32APSK | 5/6 | 54000 | 64800 | 



Receiver side: 
1. Detect SOF -> timing/freq sync
2. Demodulate PLHEADER (always PI/2-BPSK)
3. Decode PLHEADER -> extract MCS Index
4. Configured demod + LDPC decode based on MCS Index
5. Decode payload using MCS parameters
6. Compute SNR/BER -> send feedback to transmitter to adapt MCS for next frame 


How do I extract CSI from pilot symbols? 

CSI includes:
- SNR / SINR (instantaneous channel equality)
- CQI (Channel Quality Indicator) | Quantized version of SNR 
- MCS recommendation | gives index to use, e.g. MCS 12
- Channel estimate (h[k]) | complex frequency response 

Is there more? 
In DVB-S2, CSI is often simplified to Es/N0 or PER feedback since it's a broadcast link 

How Receiver Estimates CSI: 
1. Known pilot symbols are inserted at transmitter
2. Receiver compares received pilot vs known value 
    - CIR: $h(k) = \frac{y_{pilot}(k)}{x_{pilot}(k)}$
3. Average over time or frequency to get an SNR estimate
    - Ex.: $SNR_{est}=\frac{E[|x|^2]}{E[|y-hx|^2]}$
4. Optionally, smooth or filter to get a stable estimate for feedback 

How feedback is sent: 
Option A - Real 
    - CSI feedback travels over a reverse control link (uplink) 
    - In DVB-S2, it is through return channel (RCST) with Es/N0 report 
Option B - Simulated
    - Receiver compjutes SNR/PER value
    - Sent back via a ZeroMQ PUB/SUB or socket message to the transmitter process
    - Transmitter updates MCS index based on thresholds




## ACM + CSI Feedback USRP to HackRF: 
Advantage: 
- removes constraints in timing between transmit/receive
- Bandwidth: USRP up to 50MS/s, HackRF up to 20MS/s
- TX Fidelity: USRP better than HackRF
- RX Capability: HackRF sufficient for pilot/SNR estimation
- Feedback: HackRF TX -> USRP RX, very low rate (<100 kHz)
- FCC-Safe: all signals confined to cable 

```css
[USRP N210 TX] ---coax+attenuator---> [HackRF RX]
[HackRF TX] ---coax+attenuator---> [USRP N210 RX]
```

- USRP acts as forward link transmitter 
- HackRF acts as receiver and CSI estimator
- HackRF can send low-rate CSI feedback 
- No over-the-air RF emission
Note: 
- add 10-20 dB inline attenuators to avoid overloading receivers
    - HackRF RX has a higher noise floor, must adjust gain accordingly
- Use 50 $\Omega$ SMA cables
- HackRF suffers from DC spike at baseband, pilots can't be zero-frequency
- align sample rates: USRP may send at 2-10MS/s, HackRf should match exactly or use interpolation 
- use USRP clock reference for exact timing on synchronization 

Frequency Offset to Avoid DC Spike:
- HackRF DC spike / LO leakeage is from $\pm$50-100 kHz
    - transmit at 150kHz+
Optional ranges:
- 150 kHz with Rs = 1MS/s
- 200-250 kHz with Rs = 2 MS/s

Lab Setup Possibilities: 
1. Forward Link (USRP -> HackRF): 
    - BB Rs = 1-2MS/s
    - FO = 200-250 kHz
    - CF between 500Hz and 5kHz or higher 

Carrier Frequency (RF): 
- $\pm$500 Hz to $\pm$5 kHz

## Channel Estimation in DVB-S2: 
In DVB-S2, channel estimation is performed entirely at the receiver using known pilot symbols and PL (PHY Layer) framing structure. 

The transmitter does *not* measure or directly "see" the channel

### Receiver-side channel estimation process: 
#### 1. Pilot symbol positions are predefined and known to both TX and RX 
- They're insertered periodically (36 pilot symbols every 1440 data symbols) to aid coherent demodulation 
#### 2. RX uses these pilots to: 
- Estimate **channel gain and phase** (complex frequency response or impulse response) 
- Perform **equalization, phase tracking, and timing recovery**
- Optionallly compute **SNR / Es/N0 estimates** based on error between received and expected pilot symbols
#### 3. The CIR (Channel Impulse Response) or Frequency Response (H(f)) is derived from those pilots tones via correlation 

### CSI Feedback - DVB-S2 *Does Not* send Instantaneous Feedback 
Unlike 5G or Wi-Fi, DVB-S2 does not use instantaneous feedback loops. It's a **one-way (broadcast)** standard: the satellite or ground station transmits, and tens of thousands of terminals receives 

So, the receiver measures channel conditions locally, and in DVB-S2 ACM systems, the feedback path is out-of-band and slow, not instantaneous 

#### Example: 
- In interactive DVB-S2 systems (like DVB-RCS2, used for VSAT networks): 
    - the user terminal (RX) periodically sends link-quality reports (e.g. Es/N0, PER) via a separate return channel (e.g. L-band, Ku uplink) 
    - the Network Control Center (NCC) or hub modem uses that feedback to select the next MODCOD for that terminal's future downlink frames
    - Feedback latency can be hundreds of milliseconds or more, not per-frame instantaneous 
- **NOTE:** in pure broadcast DVB-S2 (TV, radio), there is no feedback at all - TX uses a fixed MODCOD for the entire multiplex

#### 4. Instaneous Channel Use - Not Feasible in DVB-S2 
Because of propagation delay (e.g. 250ms for GEO satellites), the channel response when the TX sends the next frame is already outdated  

$\therefore$ DVB-S2 never relies on instantaneous feedback, instead it relies on statistical adaptation - e.g. average Es/N0, terminal link margin, rain fade statistics 

#### 5. How Pilots are used (mathematically): 
Given some pilot symbol $P_k$ is known (e.g. BPSK = +1 or -1): 

At the RX, the corresponding received sample $Y_k$ is: 
$$Y_k = H_k P_k + N_k$$

So the channel estimate at that symbol is: 
$$H_k = \frac{Y_k}{P_K}$$

Then interpolation is performed between the pilot positions to estimate $H_n$ for all data symbols

From this, you can compute **CIR** via IFFT: 
$$h[n] = IFFT(H_k)$$

And instantaneous SNR from pilot residuals: 
$$SNR = \frac{E[|H_k P_k|^2]}{E[|Y_k - H_kP_k|^2]}$$

**This feedback interval is usually 1-10 seconds, not per frame**

### Return channel (DVB-RCS / RCS2): 
The return path will bel ower rate and on a separate link 

The receiver sends back: 
- Link metrics (SNR, Eb/N0, PER, or quantized CQI index)
- Frame acknowledgment (opt)
- Recommended MCS Index (e.g. QPSK 2/3)

This allows the transmitter to adapt:
- Modulation order 
- Code rate 
- Frame length and pilot usage 

