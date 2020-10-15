import numpy as np
import scipy as sp

# necessary functions
# import fft(Fast Fourier Transform) function to convert a signal from time domain to
from scipy.fftpack import fft
#                               frequency domain (output :is a numpy array contains signal's amplitudes of each frequency component)

# import fftfreq function to generate frequencies related to frequency components
from scipy.fftpack import fftfreq
#                                   mentioned above
# import ifft function (inverse fft) inverse the conversion
from scipy.fftpack import ifft

import math  # import math library
############################## Constants #############################
# nyq is the nyquist frequency equal to the half of the sampling frequency[50/2= 25 Hz]
sampling_freq = 50
nyq = sampling_freq / float(2)

# freq1=0.3 hertz [Hz] the cuttoff frequency between the DC compoenents [0,0.3]
freq1 = 0.3
#           and the body components[0.3,20]hz
# freq2= 20 Hz the cuttoff frequcency between the body components[0.3,20] hz
freq2 = 20
#             and the high frequency noise components [20,25] hz


# Function name: components_selection_one_signal

# Inputs: t_signal:1D numpy array (time domain signal);

# Outputs: (total_component,t_DC_component , t_body_component, t_noise)
#           type(1D array,1D array, 1D array)

# cases to discuss: if the t_signal is an acceleration signal then the t_DC_component is the gravity component [Grav_acc]
#                   if the t_signal is a gyro signal then the t_DC_component is not useful
# t_noise component is not useful
# if the t_signal is an acceleration signal then the t_body_component is the body's acceleration component [Body_acc]
# if the t_signal is a gyro signal then the t_body_component is the body's angular velocity component [Body_gyro]

def components_selection_one_signal(t_signal, freq1=freq1, freq2=freq2):
    t_signal = np.array(t_signal)
    t_signal_length = len(t_signal)  # number of points in a t_signal

    # the t_signal in frequency domain after applying fft
    f_signal = fft(t_signal)  # 1D numpy array contains complex values (in C)

    # generate frequencies associated to f_signal complex values
    # frequency values between [-25hz:+25hz]
    d = 1 / float(sampling_freq)
    freqs = np.array(sp.fftpack.fftfreq(t_signal_length, d=d))

    # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz]
    #                                                             (-0.3 and 0.3 are included)

    # noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz]
    #                                                               (-25 and 25 hz inculded 20hz and -20hz not included)

    # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz]
    #                                                               (-0.3 and 0.3 not included , -20hz and 20 hz included)

    f_DC_signal = []  # DC_component in freq domain
    f_body_signal = []  # body component in freq domain numpy.append(a, a[0])
    f_noise_signal = []  # noise in freq domain

    for i in range(len(freqs)):  # iterate over all available frequencies

        # selecting the frequency value
        freq = freqs[i]

        # selecting the f_signal value associated to freq
        value = f_signal[i]

        # Selecting DC_component values
        if abs(freq) > 0.3:  # testing if freq is outside DC_component frequency ranges
            # add 0 to  the  list if it was the case (the value should not be added)
            f_DC_signal.append(float(0))
        else:  # if freq is inside DC_component frequency ranges
            f_DC_signal.append(value)  # add f_signal value to f_DC_signal list

        # Selecting noise component values
        if (abs(freq) <= 20):  # testing if freq is outside noise frequency ranges
            # add 0 to  f_noise_signal list if it was the case
            f_noise_signal.append(float(0))
        else:  # if freq is inside noise frequency ranges
            # add f_signal value to f_noise_signal
            f_noise_signal.append(value)

        # Selecting body_component values
        # testing if freq is outside Body_component frequency ranges
        if (abs(freq) <= 0.3 or abs(freq) > 20):
            f_body_signal.append(float(0))  # add 0 to  f_body_signal list
        else:  # if freq is inside Body_component frequency ranges
            # add f_signal value to f_body_signal list
            f_body_signal.append(value)

    ################### Inverse the transformation of signals in freq domain ########################
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    _tdc = ifft(np.array(f_DC_signal))
    t_DC_component = _tdc.real
    
    _tbd = ifft(np.array(f_body_signal))
    t_body_component = _tbd.real
    
    _tns = ifft(np.array(f_noise_signal))
    t_noise = _tns.real

    # extracting the total component(filtered from noise)
    total_component = t_signal - t_noise
    #  by substracting noise from t_signal (the original signal).

    # return outputs mentioned earlier
    return (total_component, t_DC_component, t_body_component, t_noise)



from scipy.signal import butter, lfilter

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def btbp_filter_signal(sig_acc):
    # lowcut = 10.0
    lowcut = 50.0
    highcut = 1000.0
    fs = 50.0 * 128  # 128 points /(1/50) # samples(total points) / T (total time length)
    filter_raw = _butter_bandpass_filter(sig_acc, lowcut, highcut, fs, order=3)
    return filter_raw
