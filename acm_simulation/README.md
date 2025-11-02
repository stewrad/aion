# ACM Simulation Package

Complete Adaptive Coding and Modulation (ACM) simulation for satellite communications research.

## Features

✅ **Realistic Packet Generation**
- Multiple traffic types (DATA, VOICE, VIDEO, CONTROL)
- Configurable traffic mix and packet rates
- Proper packet structure with headers and CRC

✅ **Forward Error Correction (FEC)**
- Convolutional codes with multiple rates (1/2, 2/3, 3/4, 5/6)
- GNU Radio FEC integration
- Viterbi decoding

✅ **Pilot Symbol Insertion**
- Frame synchronization pilots
- MCS change notification pilots
- Periodic pilots for channel tracking
- Channel estimation from pilots

✅ **Adaptive Coding and Modulation**
- 12 MCS combinations (BPSK to 64-QAM)
- SNR-based MCS selection
- Hysteresis to prevent oscillation
- Spectral efficiency optimization

✅ **Channel Modeling**
- AWGN channel with configurable SNR
- Dynamic SNR profiles
- Channel state information (CSI)

## Package Structure

```
acm_simulation/
├── README.md                    # This file
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
│
├── acm_simulation/              # Main package
│   ├── __init__.py             # Package init
│   ├── packet_generator.py     # Packet generation
│   ├── acm_controller.py       # ACM decision logic
│   ├── pilot_inserter.py       # Pilot insertion/removal
│   ├── acm_flowgraph.py        # Main flowgraph
│   └── utils/
│       ├── __init__.py
│       └── channel_estimation.py
│
├── examples/                    # Example scripts
│   ├── run_simulation.py       # Basic simulation
│   ├── snr_sweep.py            # SNR performance test
│   └── traffic_analysis.py     # Traffic statistics
│
└── tests/                       # Unit tests
    ├── test_packet_generator.py
    ├── test_acm_controller.py
    └── test_pilot_inserter.py
```

## Installation

### Step 1: Install GNU Radio

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gnuradio python3-gnuradio
```

**Fedora:**
```bash
sudo dnf install gnuradio
```

**From source:**
See https://wiki.gnuradio.org/index.php/BuildGuide

### Step 2: Create Project Directory

```bash
cd ~/Documents
mkdir -p acm_simulation
cd acm_simulation
```

### Step 3: Save Package Files

Save each of the provided Python files into the directory:

```bash
acm_simulation/
├── __init__.py
├── packet_generator.py
├── acm_controller.py
├── pilot_inserter.py
└── acm_flowgraph.py
```

### Step 4: Create Setup Files

**requirements.txt:**
```
numpy>=1.19.0
```

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name='acm_simulation',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
    ],
    author='ACM Research',
    description='Adaptive Coding and Modulation Simulation',
    python_requires='>=3.7',
)
```

### Step 5: Install Package

```bash
cd ~/Documents/acm_simulation
pip3 install -e .
```

## Usage

### Method 1: Run as Python Script

```bash
cd ~/Documents/acm_simulation
python3 -m acm_simulation.acm_flowgraph
```

You'll see:
```
=======================================================================
 ACM SIMULATION - Adaptive Coding and Modulation 
=======================================================================

[PacketGenerator] Initialized: rate=50 pkt/s, max_size=1500 bytes
[ACMController] Initialized with 12 MCS modes
...

>>> Simulation running...

t=1s | SNR=10.0dB | MCS=QPSK | Throughput=0.85Mbps | Packets=12
t=2s | SNR=10.0dB | MCS=QPSK | Throughput=0.87Mbps | Packets=25
...
```

### Method 2: Import as Library

```python
from acm_simulation import ACMFlowgraph

# Create and run simulation
tb = ACMFlowgraph(
    snr_db=15.0,
    sample_rate=1000000,
    packet_rate=100,
    acm_enabled=True
)

tb.start()

# Change SNR dynamically
import time
for snr in [10, 15, 20, 8, 5]:
    tb.set_snr(snr)
    time.sleep(5)
    stats = tb.get_throughput()
    print(f"SNR={snr}dB: {stats['throughput_mbps']:.2f} Mbps")

tb.stop()
tb.wait()
```

### Method 3: Open in GNU Radio Companion

To create a .grc file that opens in GNU Radio Companion:

```bash
# Run the flowgraph once to generate the Python file
python3 -m acm_simulation.acm_flowgraph

# The flowgraph is now available as a Python module
# You can reference it in GRC using a Python Block
```

In GRC:
1. Add "Python Module" block
2. Set Module: `acm_simulation.acm_flowgraph`
3. Set Class: `ACMFlowgraph`
4. Add parameters as needed

## Configuration

### Adjusting ACM Parameters

Edit `acm_controller.py` to modify the ACM lookup table:

```python
ACM_TABLE = [
    # (min_snr_db, modulation, coding_rate, spectral_efficiency)
    (-2.0,  ModulationScheme.BPSK,  CodingScheme.RATE_1_2, 0.50),
    (1.0,   ModulationScheme.QPSK,  CodingScheme.RATE_1_2, 1.00),
    # Add your own MCS combinations...
]
```

### Changing Traffic Mix

```python
from acm_simulation import PacketGenerator, PacketType

packet_gen = PacketGenerator(
    traffic_mix={
        PacketType.DATA: 0.5,    # 50% data
        PacketType.VIDEO: 0.4,   # 40% video
        PacketType.VOICE: 0.1    # 10% voice
    }
)
```

### Modifying Pilot Pattern

Edit `pilot_inserter.py`:

```python
pilot_inserter = PilotInserter(
    frame_size=2048,        # Larger frames
    pilots_per_frame=128,   # More pilots for better estimation
    pilot_spacing=32        # Less frequent periodic pilots
)
```

## Running Tests

```bash
cd ~/Documents/acm_simulation
python3 -m pytest tests/
```

## Performance Analysis

### SNR Sweep Test

```python
import numpy as np
import matplotlib.pyplot as plt
from acm_simulation import ACMFlowgraph

snr_range = np.arange(0, 30, 2)
throughputs = []

for snr in snr_range:
    tb = ACMFlowgraph(snr_db=snr, acm_enabled=True)
    tb.start()
    time.sleep(10)
    
    stats = tb.get_throughput()
    throughputs.append(stats['throughput_mbps'])
    
    tb.stop()
    tb.wait()

plt.plot(snr_range, throughputs)
plt.xlabel('SNR (dB)')
plt.ylabel('Throughput (Mbps)')
plt.title('ACM Performance vs SNR')
plt.grid()
plt.savefig('acm_performance.png')
```

### Comparing ACM vs Fixed MCS

```python
# Run with ACM
tb_acm = ACMFlowgraph(snr_db=10, acm_enabled=True)
# ... collect statistics ...

# Run with fixed QPSK
tb_fixed = ACMFlowgraph(snr_db=10, acm_enabled=False)
# ... collect statistics ...

# Compare throughput, packet loss, etc.
```

## Troubleshooting

### "ImportError: cannot import name 'ACMFlowgraph'"

**Fix:** Make sure you're in the correct directory and the package is installed:
```bash
cd ~/Documents/acm_simulation
pip3 install -e .
```

### "RuntimeError: can't open file"

**Fix:** The output directory doesn't exist:
```bash
mkdir -p /tmp
# Or change output path in acm_flowgraph.py
```

### "AttributeError: 'module' object has no attribute 'constellation_64qam'"

**Fix:** Your GNU Radio version doesn't have 64-QAM. Either:
- Upgrade GNU Radio to 3.9+
- Comment out 64-QAM from the ACM table in `acm_controller.py`

### Low throughput or high packet loss

**Check:**
- SNR is appropriate for selected MCS
- FEC frame size matches packet size
- Throttle block is present and configured correctly

## Next Steps

### Adding Reinforcement Learning

To add RL for ACM+RL comparison:

1. Create `rl_agent.py` module
2. Replace ACM controller decision logic with RL agent
3. Train agent using reward = throughput × (1 - packet_loss_rate)
4. Compare performance

### Adding More Realistic Channel

Replace simple AWGN with:
- Rician/Rayleigh fading
- Doppler shift
- Multipath propagation

Edit in `acm_flowgraph.py`:
```python
self.channel = channels.channel_model(
    noise_voltage=noise_voltage,
    frequency_offset=1000.0,  # Doppler
    epsilon=1.0,
    taps=[1.0, 0.5, 0.3],     # Multipath
    noise_seed=0
)
```

### Logging for Analysis

```python
import csv

with open('acm_log.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'snr', 'mcs', 'throughput', 'packets'])
    
    # During simulation
    writer.writerow([time.time(), snr, mcs_info['modulation'], 
                    throughput['throughput_mbps'], 
                    packet_stats['packets_generated']])
```

## References

- DVB-S2 Standard: ETSI EN 302 307
- GNU Radio Documentation: https://wiki.gnuradio.org
- Adaptive Modulation and Coding: A. Goldsmith, "Wireless Communications"

## License

MIT License - Feel free to use for research and education.

## Contact

For questions or issues, please open an issue on the project repository.