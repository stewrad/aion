import numpy as np

# -----------------------
# APSK constellation definitions
# -----------------------
def get_apsk_constellation(mod):
    if mod=='8APSK':
        r1, r2 = 1, 2
        angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        points = np.concatenate([r1*np.exp(1j*angles1), r2*np.exp(1j*angles2)])
    elif mod=='16APSK':
        r1, r2 = 1, 2
        points = np.concatenate([r1*np.exp(1j*np.linspace(0,2*np.pi,4,endpoint=False)),
                                 r2*np.exp(1j*np.linspace(0,2*np.pi,12,endpoint=False))])
    elif mod=='32APSK':
        r1, r2, r3 = 1, 2, 3
        points = np.concatenate([r1*np.exp(1j*np.linspace(0,2*np.pi,4,endpoint=False)),
                                 r2*np.exp(1j*np.linspace(0,2*np.pi,12,endpoint=False)),
                                 r3*np.exp(1j*np.linspace(0,2*np.pi,16,endpoint=False))])
    else:
        return None
    return points