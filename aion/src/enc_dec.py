import numpy as np
from sionna.phy.fec import utils
import scipy.sparse as sp

# -------------------------
# Load DVB-S2 H from ALIST
# -------------------------
alist_file = "alist/dvbs2_1_2_N16200.alist"
alist = utils.load_alist(alist_file)
H_dense, _, N, _ = utils.alist2mat(alist)
M = H_dense.shape[0]
K = N - M
H_sparse = sp.csr_matrix(H_dense, dtype=np.uint8)
# print(f"Loaded H: shape={H_sparse.shape}, K={K}, N={N}")

# -------------------------
# Convert H to systematic form to compute G
# -------------------------
def compute_generator_matrix(H):
    """Convert H to systematic form [P^T | I] and compute G = [I | P]"""
    H = H.copy().astype(int)
    M, N = H.shape
    K = N - M

    H_sys = H.copy()
    pivot_cols = []
    for row in range(M):
        # find pivot in columns K..N-1
        for col in range(K, N):
            if H_sys[row, col] == 1:
                pivot_cols.append(col)
                if col != K + row:
                    H_sys[:, [col, K + row]] = H_sys[:, [K + row, col]]
                break
        else:
            raise RuntimeError("Failed to find pivot for systematic form")
        # eliminate ones above and below
        for r in range(M):
            if r != row and H_sys[r, K + row] == 1:
                H_sys[r, :] = (H_sys[r, :] + H_sys[row, :]) % 2

    # H_sys = [A | I_M]
    A = H_sys[:, :K]
    G = np.concatenate([np.eye(K, dtype=int), A.T % 2], axis=1)
    return G

# -------------------------
# Compute G
# -------------------------
G = compute_generator_matrix(H_dense)
# print("Generator matrix computed:", G.shape)

# -------------------------
# Encode info bits
# -------------------------
u_bits = np.random.randint(0, 2, K, dtype=np.uint8)
codeword = (u_bits @ G % 2).astype(np.uint8)
# print("  Encoding complete")
# print("Codeword length:", len(codeword))

# -------------------------
# Parity check
# -------------------------
parity_check = H_sparse.dot(codeword) % 2
if np.all(parity_check == 0):
    print("  Parity check passed")
else: 
    print(f"  Parity check failed: {np.sum(parity_check != 0)} unsatisfied parity checks")

# -------------------------
# Codeword ready for modulation
# -------------------------
# Convert to float if needed for BPSK/QPSK mapping
# mod_bits = 2*codeword.astype(np.float32) - 1  # BPSK: 0->-1, 1->+1
# print("Modulation-ready bits:", mod_bits[:32], "...")  # preview first 32 bits
mod_input_bits = codeword.copy()
# print("First 32 encoded bits:", mod_input_bits[:32])
# Example reshape for QPSK
# symbols = mod_input_bits.reshape(-1, 2)  # for QPSK

# -------------------------
# Optional: simple bit-flipping decoder
# -------------------------
def bit_flipping_decode(cw, H_sparse, max_iter=50):
    c = cw.copy()
    H = H_sparse.tolil()
    for _ in range(max_iter):
        synd = H.dot(c) % 2
        if np.all(synd == 0):
            break
        unsat = synd.nonzero()[0]
        flip_counts = np.zeros(c.shape[0], dtype=int)
        for chk in unsat:
            cols = H.rows[chk]
            flip_counts[cols] += 1
        to_flip = np.where(flip_counts > (len(H.rows[0])//2))[0]
        c[to_flip] ^= 1
    return c

decoded = bit_flipping_decode(codeword, H_sparse)
errors = np.sum(decoded[:K] != u_bits)
# print(f"Info bits recovered errors: {errors}")





# # Possible Decoder following Demodulation in simulation:
# from sionna.fec.ldpc.encoding import LDPC5GEncoder
# from sionna.fec.ldpc.decoding import LDPC5GDecoder

# encoder = LDPC5GEncoder(H_sparse)
# decoder = LDPC5GDecoder(H_sparse, hard_out=True, num_iter=50)

# # Encode and decode
# codeword = encoder(u_bits)
# decoded_bits = decoder(llr_inputs)
