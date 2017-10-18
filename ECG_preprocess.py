'''
HRV=IBI=RR interval
HR~=1/HRV 123456
'''
# IBI's sampling rate =heart rate#
'''
Root mean square of the mean squared of IBIs
 mean IBI
60 spectral power in the bands from [0-6] Hz component of the ECG signal
low frequency [0.01,0.08]Hz, medium frequency
[0.08,0.15] and hight frequency [0.15,0.5] Hz components of
HRV spectral power
 HR and HRV stats.
'''
import _pickle as cpickle
import numpy as np
from biosppy.signals import ecg
from biosppy.signals import eeg
#################################
def filter(spectrum,lower,upper):
	#filtered=[]
	#filtered=np.array(filtered)
	lo_idx	= (np.abs(spectrum-lower)).argmin()
	up_idx	= (np.abs(spectrum-upper)).argmin()
	'''
	for x in np.nditer(spectrum):
		if((x<upper)and(x>lower)):
			filtered=np.append(filtered, x)
		elif (x>upper):
			break
			'''
	print('lo_idx.',lo_idx)
	print('up_idx.',up_idx)
	return [lo_idx,up_idx]
#################################
def power(spectrum,idx_pairs):
	lo_idx=idx_pairs[0]
	up_idx=idx_pairs[1]
	power=0
	spectrumbuf=spectrum[lo_idx:up_idx+1]#???????????
	for x in np.nditer(spectrumbuf):
		power+=abs(x*x)
	print('np.array([power]).')
	print(np.array([power]))
	return np.array([power])
	
#################################
signal = np.genfromtxt('1_1.csv',delimiter=',')
ecg_signal=signal[:,14:15]
ecg_signal=np.array(ecg_signal)


ecg_signal=np.reshape(ecg_signal,(ecg_signal.shape[0],))
print(ecg_signal.shape)
#####################################
ecg_all = ecg.ecg(signal=ecg_signal, sampling_rate=128., show=False)
timestep = 1/128


##########################################
ts =ecg_all['ts']
filtered =ecg_all['filtered']
rpeaks=ecg_all['rpeaks']
templates_ts=ecg_all['templates_ts']
templates=ecg_all['templates']
heart_rate_ts =ecg_all['heart_rate_ts']
heart_rate =ecg_all['heart_rate']
#################fft of raw data############
ecg_fourier= np.fft.fft(ecg_signal)
 
ecg_freq_idx = np.fft.fftfreq(ecg_signal.size, d=1/128)
#d : scalar, optional Sample spacing (inverse of the sampling rate)
positive_ecg_freq_idx=ecg_freq_idx[:(int((ecg_freq_idx.shape[0]+1)/2))]
print('positive_ecg_freq_idx',positive_ecg_freq_idx)

########################################
power_0_6=np.array([])
for i in range(60):
	power_0_6=np.append(power_0_6,power(ecg_fourier,\
	(filter(positive_ecg_freq_idx,0+(i*0.1),0.1+(i*0.1)))))
###################60feature############################

	

print('power_0_6.shape',power_0_6.shape[0])

print('power_0_6.',power_0_6)

##############IBI#################
IBI=[]
print('rpeaks',rpeaks)
for i in range(len(rpeaks)-1):
	IBI.append(rpeaks[i+1]-rpeaks[i])

IBI=np.array(IBI)
mean_IBI=np.mean(IBI)#####feature--IBI stats#####	
print('mean_IBImean_IBImean_IBImean_IBImean_IBImean_IBI',mean_IBI)
HR=128/mean_IBI*60#########feature--HR stats(bpm)##########

rms_IBI = np.sqrt(np.mean(np.square(IBI)))#####feature--rms_IBI#####	
###############fft of raw IBI
IBI_fourier= np.fft.fft(IBI)
IBI_freq_idx = np.fft.fftfreq(IBI.size, d=(128/mean_IBI))
positive_IBI_freq_idx=IBI_freq_idx[:(int((IBI_freq_idx.shape[0]+1)/2))]
####/2)+1) round to interger!!##


#print('IBI_freq_idx',IBI_freq_idx)
#print('len(IBI_freq_idx)',IBI_freq_idx.shape)

#print('positive_IBI_freq_idx',positive_IBI_freq_idx)
#print('len(positive_IBI_freq_idx)',positive_IBI_freq_idx.shape)

###########feature[0.01,0.08]Hz##################
power_001_008=np.array([])
power_001_008=np.append(power_001_008,power(IBI_fourier,np.array\
(filter(positive_IBI_freq_idx,0.01,0.08))))
###########feature[0.08,0.15]Hz##################
power_008_015=np.array([])
power_008_015=np.append(power_008_015,power(IBI_fourier,np.array\
(filter(positive_IBI_freq_idx,0.08,0.15))))	
###########feature[0.15,0.5]Hz##################
power_015_05=np.array([])
power_015_05=np.append(power_015_05,power(IBI_fourier,np.array\
(filter(positive_IBI_freq_idx,0.15,0.5))))

###############################################save files####################
feature_dict={'rms_IBI':rms_IBI,'mean_IBI':mean_IBI,'HR':HR,'power_0_6':power_0_6,\
'power_001_008':power_001_008,'power_008_015':power_008_015,\
'power_015_05':power_015_05}
f = open("./ecgfeature.txt","wb")
cpickle.dump(feature_dict, f)
#print('power_008_015',power_008_015)
#print('power_015_05',power_015_05)
