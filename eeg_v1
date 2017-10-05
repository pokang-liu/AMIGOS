import numpy as np
from biosppy.signals import ecg
from biosppy.signals import eeg
# load raw signal
signal = np.genfromtxt('1_1.csv',delimiter=',')



eeg_signal=signal[:,:14]
ecg_signal=signal[:,14:16]
gsm_signal=signal[:,-1]
print('signal.shape',signal.shape)
print('eeg_signal.shape',eeg_signal.shape)
print('ecg_signal.shape',ecg_signal.shape)
print('gsm_signal.shape',gsm_signal.shape)

####################
eeg_all = eeg.eeg(signal=eeg_signal, sampling_rate=128., show=False)
#ecg_all = ecg.ecg(signal=eeg_signal, sampling_rate=128., show=False)
print(eeg_all['beta'])

ts =eeg_all['ts']
filtered =eeg_all['filtered']
features_ts=eeg_all['features_ts']
theta=eeg_all['theta']
alpha_low =eeg_all['alpha_low']
alpha_high =eeg_all['alpha_high']
beta =eeg_all['beta']
gamma =eeg_all['gamma']
plf_pairs =eeg_all['plf_pairs']
plf =eeg_all['plf']
beta=eeg_all['beta']
####################
theta_power=np.sum(theta,axis=0)
print('theta',theta)
alpha_low_power=np.sum(alpha_low,axis=0)
print('alpha_low',alpha_low)
alpha_high_power=np.sum(alpha_high,axis=0)
print('alpha_high',alpha_high)
beta_power=np.sum(beta,axis=0)
print('beta',beta)
gamma_power=np.sum(gamma,axis=0)

'''
print('theta_power',theta_power)
print('theta_power.shape[1]',theta_power.shape[1])
'''
theta_spa=[]
alpha_low_spa=[]
alpha_high_spa=[]
beta_spa=[]
gamma_spa=[]

for i in range(7):
	theta_spa.append((theta_power[i]-theta_power[13-i])/\
	(theta_power[i]+theta_power[13-i]))
	
	alpha_low_spa.append((alpha_low_power[i]-alpha_low_power[13-i])/\
	(alpha_low_power[i]+alpha_low_power[13-i]))
	
	alpha_high_spa.append((alpha_high_power[i]-alpha_high_power[13-i])/\
	(alpha_high_power[i]+alpha_high_power[13-i]))
	
	beta_spa.append((beta_power[i]-beta_power[13-i])/\
	(beta_power[i]+beta_power[13-i]))
	
	gamma_spa.append((gamma_power[i]-gamma_power[13-i])/\
	(gamma_power[i]+gamma_power[13-i]))
	
print('theta_spa',theta_spa)

####save the data
eep_power=[]
eeg_spa=[]
power_name={1:theta_power,2:alpha_low_power,3:alpha_high_power\
,4:beta_power,5:gamma_power}

spa_name={1:theta_spa,2:alpha_low_spa,3:alpha_high_spa\
,4:beta_spa,5:gamma_spa}
for i in spa_name:
	eeg_spa.append(spa_name[i])
	
for i in power_name:
	eep_power.append(power_name[i])

eeg_power=np.array(eep_power)
eeg_spa=np.array(eeg_spa)

print('eeg_spa.shape',eeg_spa.shape)
print('eeg_power.shape',eeg_power.shape)	


print('eeg_spa',eeg_spa)
print('eeg_power',eeg_power)


np.save('eep_power',eep_power)
np.save('eeg_spa',eeg_spa)
'''
data format:

eeg_spa.npy

channel number(0~7)
theta_spa
alpha_low_spa
alpha_high_spa
beta_spa
gamma_spa

eep_power.npy
channel number(0~13)
theta_power
alpha_low_power
alpha_high_power
beta_power
gamma_power
'''
