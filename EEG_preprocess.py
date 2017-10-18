import sys 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

SAMPLE_RATE = 128
	
def find_freq_interval(f, Pxx_den,lowerbound,upperbound):
	Pxx_den_ 	= np.array(Pxx_den)
	low_idx 	= np.where(f >= lowerbound) # np.where return tuple
	low_idx		= np.array(low_idx).flatten()
	high_idx 	= np.where(f >= upperbound) 
	high_idx	= np.array(high_idx).flatten()
	Pxx_den_[:low_idx[0]] = 0;
	Pxx_den_[high_idx[0]:] = 0;
	
	Pxx_den_mean = np.sum(Pxx_den_)/(high_idx[0]-low_idx[0])
	return Pxx_den_mean
	
def compute_PSD(data, fs, nperseg):
	data = np.transpose(data) # each channel in a row
	
	theta_power 		= np.zeros(data.shape[0])
	slow_alpha_power 	= np.zeros(data.shape[0])
	alpha_power 		= np.zeros(data.shape[0])
	beta_power 			= np.zeros(data.shape[0])
	gamma_power 		= np.zeros(data.shape[0])
	
	for i in range(data.shape[0]):
		f , Pxx_den = signal.welch(data[i], fs, nperseg = nperseg)
		theta_power[i] 		= find_freq_interval(f, Pxx_den, 3, 7)
		slow_alpha_power[i] = find_freq_interval(f, Pxx_den, 8, 10)
		alpha_power[i] 		= find_freq_interval(f, Pxx_den, 8, 13)
		beta_power[i]		= find_freq_interval(f, Pxx_den, 14, 29)
		gamma_power[i]		= find_freq_interval(f, Pxx_den, 30, 47)

	return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power

def asymmetry(power):
	asy_power = np.zeros(power.shape[0]/2)
	for i in range(asy_power.shape[0]):
		asy_power[i] = power[i]-power[13-i]
	
	return asy_power

def eeg_extract(eeg_signal):
	theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power= compute_PSD(eeg_signal, SAMPLE_RATE, nperseg=128)
	
	asy_theta_power 		= asymmetry(theta_power)
	asy_slow_alpha_power	= asymmetry(slow_alpha_power)
	asy_alpha_power 		= asymmetry(alpha_power)
	asy_beta_power 			= asymmetry(beta_power)
	asy_gamma_power 		= asymmetry(gamma_power)

	eeg_features = {
        'theta_power': theta_power,
        'slow_alpha_power': slow_alpha_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'asy_theta_power': asy_theta_power,
        'asy_slow_alpha_power': asy_slow_alpha_power,
        'asy_alpha_power': asy_alpha_power,
        'asy_beta_power': asy_beta_power,
        'asy_gamma_power': asy_gamma_power
    }
	
	return eeg_features
	
if __name__ == "__main__":
	filename = './AMIGOS_data/' + sys.argv[1] +'.csv'
	AMIGOS_data = np.genfromtxt(filename, delimiter=',')
	
	EEG_signal = AMIGOS_data[:,:14] 
	ECG_signal = AMIGOS_data[:,14] #pick one of two ECG signals
	GSR_signal = AMIGOS_data[:,-1]
	
	EEG_features = eeg_extract(EEG_signal)
	# ECG_features = ecg_extract(ECG_signal)
	# GSR_features = gsr_extract(GSR_signal)
	
	for keys,values in EEG_features.items():
		print(keys)
		print(values)

	


