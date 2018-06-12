import numpy as np

old = np.loadtxt('data/features.csv', delimiter=',')[:, :213]
new = np.loadtxt('data/mse_mpe_features.csv', delimiter=',')

a_mse = new[:, 213:229]
v_mse = new[:, 229:233]

a_eeg_mmpe = new[:, 233:236]
v_eeg_mmpe = new[:, 236:325]
a_ecg_rcmpe = new[:, 325:327]
v_ecg_rcmpe = new[:, 327:335]
a_gsr_rcmpe = new[:, 335:355]

a_mpe = np.hstack((a_eeg_mmpe, a_ecg_rcmpe, a_gsr_rcmpe))
v_mpe = np.hstack((v_eeg_mmpe, v_ecg_rcmpe))

np.savetxt('data/a_features.csv', np.hstack((old, a_mse, a_mpe)), delimiter=',')
np.savetxt('data/v_features.csv', np.hstack((old, v_mse, v_mpe)), delimiter=',')
