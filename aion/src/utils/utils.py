# from astrol.config import load_config, config
import numpy as np
import warnings

from typing import Union

def str2arr(
    inString: str
):
    oArray = np.array(list(inString), dtype=int)

    return oArray

def bin2dibit(
    bits: np.ndarray
):
    '''
    Convert input binary array into dibit format, padding with zeros as necessary  
    '''
    # Ensure the binary array has an even length
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
        
    # Reshape the array into pairs of bits
    reshaped_array = bits.reshape(-1, 2)
    
    # Convert each pair of bits into a dibit (as an integer)
    dibit_array = reshaped_array[:, 0] * 2 + reshaped_array[:, 1]
    
    return dibit_array

def dibit2bit(
    dibits: np.ndarray
):
    '''
    Convert input dibit array back into binary array
    '''
    # Convert each dibit to a 2-bit binary string and concatenate them
    bit_array = np.array([list(format(d, '02b')) for d in dibits])
    
    # Tranpose the 2D array to get a 1D array of binary bits
    bit_array = bit_array.flatten().astype(int)
    
    return bit_array

def bin2octal(
    bits_arr: np.ndarray
):
    '''
    Convert input binary array into octal format, padding with zeros as necessary 
    '''
    # Ensure the binary array has an even length
    if len(bits_arr) % 3 != 0:
        if len(bits_arr) % 3 == 1:
            bits_arr = np.pad(bits_arr, (0, 2))
        elif len(bits_arr) % 3 == 2:
            bits_arr = np.pad(bits_arr, (0, 1))
    
    # Reshape the array into pairs of bits
    reshaped_array = bits_arr.reshape(-1, 3)
    
    # Convert each pair of bits into a octal (as an integer)
    oct_array = reshaped_array[:, 0] * 4 + reshaped_array[:, 1] * 2 + reshaped_array[:, 2]
    
    return oct_array

def hex2bin(
    hex_str: str      
):
    '''
    Convert input hexadecimal string into np.ndarray of binary values, padding with zeros for left-zero padding (e.g. 0x001a)
    '''
    bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)

    oArray = np.array(list(bin_str), dtype=int)

    return oArray

def bin2hex_str(
    bin_str: str      
):
    '''
    Convert binary string into hexadecimal string 
    '''
    hex_str = '%0*X' % ((len(bin_str) + 3) // 4, int(bin_str, 2))

    return hex_str

def bin2decimal(bits_2d: np.ndarray) -> np.ndarray:
    """
    Convert 2D array of binary bits to decimal values.
    
    Parameters
    ----------
    bits_2d : np.ndarray of shape (n_symbols, bits_per_symbol)
        Binary bits arranged as rows of symbols
    
    Returns
    -------
    np.ndarray of shape (n_symbols,)
        Decimal values for each symbol
    """
    bits_per_symbol = bits_2d.shape[1]
    powers = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    return np.dot(bits_2d, powers)