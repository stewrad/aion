import zmq
import numpy as np

ZMQ_ADDR = "tcp://127.0.0.1:5555"
ctx = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect(ZMQ_ADDR)
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print(f"RX SUB connected to {ZMQ_ADDR}")

while True:
    data_bytes = socket.recv()
    samples = np.frombuffer(data_bytes, dtype=np.complex64)
    
    # ====== Example processing ======
    # Compute simple energy metric to detect bursts
    energy = np.mean(np.abs(samples)**2)
    if energy > 0.1:  # threshold for burst detection
        print(f"Detected burst: Energy={energy:.3f}, Length={len(samples)}")
        
        # TODO: Extract pilots → estimate CIR
        # cir = estimate_cir(samples, pilot_pattern)
        
    # Forward to GRC or further processing
