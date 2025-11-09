from matplotlib import pylab as plt
import numpy as np 

def plot_all(
    inSig: np.ndarray,
    Fs: float,
    Rs: float, 
    sps: float, 
):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs = axs.flatten()

    # 1. Spectrogram of inSig
    axs[0].specgram(inSig, NFFT=256, Fs=Fs, noverlap=128, scale='dB')
    axs[0].set_title("Spectrogram")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Frequency [Hz]")

    # 2. Constellation
    axs[1].plot(np.real(inSig[::sps]), np.imag(inSig[::sps]), '.-', color='red', markerfacecolor='blue', markeredgecolor='blue', markersize=0.5)
    axs[1].set_title("Constellation")
    axs[1].set_xlabel("In-Phase")
    axs[1].set_ylabel("Quadrature")
    axs[1].axis('equal')

    plt.tight_layout()
    plt.show()

    # PLOT_DIR = fPath 
    # os.makedirs(config.PLOT_DIR, exist_ok=True)
    # plot_path = config.PLOT_DIR / fName 
    # plot_filename = Path(str(plot_path) + '.png')
    # fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

    # plt.close(fig)
