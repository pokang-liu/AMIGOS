import sys 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def avg_ref(data):
	mean 	= np.mean(data, axis=0)
	data 	= data - mean
	return data
	
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
	b, a = butter_highpass(cutoff, fs, order=order)

	for i in range(data.shape[1]):
		data[:,i] = signal.filtfilt(b, a, data[:,i])
	return data
	
def find_freq_interval(f, Pxx_den,lowerbound,upperbound):
	Pxx_den_ = Pxx_den[:]
	for i in range(f.shape[1]):
		low_idx = np.where(f[:,i]>=lowerbound) # low[0] 
		high_idx = np.where(f[:,i]>upperbound) # high[0]
		Pxx_den_[0:low_idx,i] = 0;
		Pxx_den_[high_idx,:] = 0;
	
	Pxx_den_mean = np.mean(Pxx_den_, axis=0)
	return Pxx_den_mean
	
def compute_PSD(data, fs, nperseg = 128):
	f 		= np.zeros(data.shape)
	Pxx_den = np.zeros(data.shape)
	for i in range(data.shape[1]):
		f[:,i] , Pxx_den[:,i] = signal.welch(data[:,i], fs, nperseg)
	theta_energy 		= find_freq_interval(f, Pxx_den, 3, 7)
	slow_alpha_energy 	= find_freq_interval(f, Pxx_den, 8, 10)
	alpha_energy 		= find_freq_interval(f, Pxx_den, 8, 13)
	beta_energy			= find_freq_interval(f, Pxx_den, 14, 29)
	gamma_energy 		= find_freq_interval(f, Pxx_den, 30, 47)

	return theta_energy, slow_alpha_energy, alpha_energy, beta_energy, gamma_energy

############################# Load data
filename = './AMIGOS_data/' + sys.argv[1] +'.csv'
original_EEG_signal = np.delete(np.genfromtxt(filename, delimiter=','),[14,15,16],1)

############################# pre-processing
#1. average-referenced
avg_EEG_signal = avg_ref(original_EEG_signal)

#2. HP filter with fc = 2 Hz
filtered_EEG_signal = butter_highpass_filter(avg_EEG_signal, 2 , 128)

############################ extract features
theta_energy, slow_alpha_energy, alpha_energy, beta_energy, gamma_energy = compute_PSD(filtered_EEG_signal,128)

asymmetry = np.zeros((7,5))
for i in range(7):
	asymmetry[i] = [theta_energy[i+1]-theta_energy[14-i],	slow_alpha_energy[i+1]-slow_alpha_energy[14-i],	alpha_energy[i+1]-alpha_energy[14-i],	beta_energy[i+1]-beta_energy[14-i],	gamma_energy[i+1]-gamma_energy[14-i]]
	
############################# Plot
plt.figure(figsize=(20,10))
plt.subplot(311)
plt.plot(original_EEG_signal[:,0])
plt.subplot(312)
plt.plot(avg_EEG_signal[:,0])
plt.subplot(313)
plt.plot(filtered_EEG_signal[:,0])
plt.show()



#Ref: https://stackoverflow.com/questions/39032325/python-high-pass-filter
#Ref: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html



