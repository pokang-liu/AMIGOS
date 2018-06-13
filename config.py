#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Configuration (Global variables)
'''

SUBJECT_NUM = 40
VIDEO_NUM = 16
SAMPLE_RATE = 128.
MISSING_DATA_SUBJECT = [9, 12, 21, 22, 23, 24, 33]
FEATURE_NAMES = ['mean_SR', 'mean_deri', 'mean_dif_neg', 'propo_neg', 'num_local_min', 'mean_rise', 'sp_0.0_0.2', 'sp_0.2_0.4', 'sp_0.4_0.6', 'sp_0.6_0.8', 'sp_0.8_1.0', 'sp_1.0_1.2', 'sp_1.2_1.4', 'sp_1.4_1.6', 'sp_1.6_1.8', 'sp_1.8_2.0', 'sp_2.0_2.2', 'sp_2.2_2.4', 'mean_SC', 'std_SC', 'mean_der1_SC', 'mean_der2_SC', 'mean_SCSR', 'std_SCSR', 'mean_der1_SCSR', 'mean_der2_SCSR', 'ZC_SCSR', 'ZC_SCVSR', 'mag_SCSR', 'mag_SCVSR', 'mag_all', 'occ_ratio', 'rms_pin', 'mean_pin', 'std_pin', 'ske_pin', 'kur_pin', 'usum_pin', 'dsum_pin', 'mean_HR', 'std_HR', 'ske_HR', 'kur_HR', 'usum_HR', 'dsum_HR', 'low_HRV_fre', 'mid_HRV_fre', 'high_HRV_fre', 'ecg_sp_0.0_0.1', 'ecg_sp_0.1_0.2', 'ecg_sp_0.2_0.3', 'ecg_sp_0.3_0.4', 'ecg_sp_0.4_0.5', 'ecg_sp_0.5_0.6', 'ecg_sp_0.6_0.7', 'ecg_sp_0.7_0.8', 'ecg_sp_0.8_0.9', 'ecg_sp_0.9_1.0', 'ecg_sp_1.0_1.1', 'ecg_sp_1.1_1.2', 'ecg_sp_1.2_1.3', 'ecg_sp_1.3_1.4', 'ecg_sp_1.4_1.5', 'ecg_sp_1.5_1.6', 'ecg_sp_1.6_1.7', 'ecg_sp_1.7_1.8', 'ecg_sp_1.8_1.9', 'ecg_sp_1.9_2.0', 'ecg_sp_2.0_2.1', 'ecg_sp_2.1_2.2', 'ecg_sp_2.2_2.3', 'ecg_sp_2.3_2.4', 'ecg_sp_2.4_2.5', 'ecg_sp_2.5_2.6', 'ecg_sp_2.6_2.7', 'ecg_sp_2.7_2.8', 'ecg_sp_2.8_2.9', 'ecg_sp_2.9_3.0', 'ecg_sp_3.0_3.1', 'ecg_sp_3.1_3.2', 'ecg_sp_3.2_3.3', 'ecg_sp_3.3_3.4', 'ecg_sp_3.4_3.5', 'ecg_sp_3.5_3.6', 'ecg_sp_3.6_3.7', 'ecg_sp_3.7_3.8', 'ecg_sp_3.8_3.9', 'ecg_sp_3.9_4.0', 'ecg_sp_4.0_4.1', 'ecg_sp_4.1_4.2', 'ecg_sp_4.2_4.3', 'ecg_sp_4.3_4.4', 'ecg_sp_4.4_4.5', 'ecg_sp_4.5_4.6', 'ecg_sp_4.6_4.7', 'ecg_sp_4.7_4.8', 'ecg_sp_4.8_4.9', 'ecg_sp_4.9_5.0', 'ecg_sp_5.0_5.1', 'ecg_sp_5.1_5.2', 'ecg_sp_5.2_5.3', 'ecg_sp_5.3_5.4', 'ecg_sp_5.4_5.5', 'ecg_sp_5.5_5.6', 'ecg_sp_5.6_5.7', 'ecg_sp_5.7_5.8', 'ecg_sp_5.8_5.9', 'ecg_sp_5.9_6.0', 'theta_AF3', 'slow_alpha_AF3', 'alpha_AF3', 'beta_AF3', 'gamma_AF3', 'theta_AF4', 'slow_alpha_AF4', 'alpha_AF4', 'beta_AF4', 'gamma_AF4', 'theta_AF3_AF4', 'slow_alpha_AF3_AF4', 'alpha_AF3_AF4', 'beta_AF3_AF4', 'gamma_AF3_AF4', 'theta_F7', 'slow_alpha_F7', 'alpha_F7', 'beta_F7', 'gamma_F7', 'theta_F8', 'slow_alpha_F8', 'alpha_F8', 'beta_F8', 'gamma_F8', 'theta_F7_F8', 'slow_alpha_F7_F8', 'alpha_F7_F8', 'beta_F7_F8', 'gamma_F7_F8', 'theta_F3', 'slow_alpha_F3', 'alpha_F3', 'beta_F3', 'gamma_F3', 'theta_F4', 'slow_alpha_F4', 'alpha_F4', 'beta_F4', 'gamma_F4', 'theta_F3_F4', 'slow_alpha_F3_F4', 'alpha_F3_F4', 'beta_F3_F4', 'gamma_F3_F4', 'theta_FC5', 'slow_alpha_FC5', 'alpha_FC5', 'beta_FC5', 'gamma_FC5', 'theta_FC6', 'slow_alpha_FC6', 'alpha_FC6', 'beta_FC6', 'gamma_FC6', 'theta_FC5_FC6', 'slow_alpha_FC5_FC6', 'alpha_FC5_FC6', 'beta_FC5_FC6', 'gamma_FC5_FC6', 'theta_T7', 'slow_alpha_T7', 'alpha_T7', 'beta_T7', 'gamma_T7', 'theta_T8', 'slow_alpha_T8', 'alpha_T8', 'beta_T8', 'gamma_T8', 'theta_T7_T8', 'slow_alpha_T7_T8', 'alpha_T7_T8', 'beta_T7_T8', 'gamma_T7_T8', 'theta_P7', 'slow_alpha_P7', 'alpha_P7', 'beta_P7', 'gamma_P7', 'theta_P8', 'slow_alpha_P8', 'alpha_P8', 'beta_P8', 'gamma_P8', 'theta_P7_P8', 'slow_alpha_P7_P8', 'alpha_P7_P8', 'beta_P7_P8', 'gamma_P7_P8', 'theta_O1', 'slow_alpha_O1', 'alpha_O1', 'beta_O1', 'gamma_O1', 'theta_O2', 'slow_alpha_O2', 'alpha_O2', 'beta_O2', 'gamma_O2', 'theta_O1_O2', 'slow_alpha_O1_O2', 'alpha_O1_O2', 'beta_O1_O2', 'gamma_O1_O2']
A_FEATURE_NAMES = ['ecg_rcmse_m0_s2','ecg_rcmse_m1_s2','gsr_rcmse_7','gsr_rcmse_8','gsr_rcmse_9','gsr_rcmse_10','gsr_rcmse_11','gsr_rcmse_12','gsr_rcmse_13','gsr_rcmse_14','gsr_rcmse_15','gsr_rcmse_16','gsr_rcmse_17','gsr_rcmse_18','gsr_rcmse_19','gsr_rcmse_20', 'eeg_mmpe_s18_d2_r3', 'eeg_mmpe_s19_d4_r1', 'eeg_mmpe_s11_d2_r4', 'ecg_rcmpe_s2_d3', 'ecg_rcmpe_s1_d3', 'gsr_rcmpe_s20_d2', 'gsr_rcmpe_s19_d2', 'gsr_rcmpe_s18_d2', 'gsr_rcmpe_s17_d2', 'gsr_rcmpe_s16_d2', 'gsr_rcmpe_s15_d2', 'gsr_rcmpe_s14_d2', 'gsr_rcmpe_s13_d2', 'gsr_rcmpe_s12_d2', 'gsr_rcmpe_s11_d2', 'gsr_rcmpe_s10_d2', 'gsr_rcmpe_s9_d2', 'gsr_rcmpe_s8_d2', 'gsr_rcmpe_s7_d2', 'gsr_rcmpe_s6_d2', 'gsr_rcmpe_s1_d2', 'gsr_rcmpe_s5_d2', 'gsr_rcmpe_s4_d2', 'gsr_rcmpe_s3_d2', 'gsr_rcmpe_s1_d3']
V_FEATURE_NAMES = ['ecg_rcmse_m0_s2','ecg_rcmse_m1_s2','ecg_rcmse_m2a_s2','ecg_rcmse_m0_s3', 'eeg_mmpe_s17_d6_r2', 'eeg_mmpe_s20_d6_r2', 'eeg_mmpe_s20_d6_r4', 'eeg_mmpe_s18_d6_r5', 'eeg_mmpe_s14_d6_r2', 'eeg_mmpe_s15_d6_r2', 'eeg_mmpe_s18_d6_r2', 'eeg_mmpe_s18_d6_r3', 'eeg_mmpe_s17_d6_r3', 'eeg_mmpe_s15_d6_r5', 'eeg_mmpe_s14_d6_r3', 'eeg_mmpe_s20_d6_r1', 'eeg_mmpe_s16_d6_r2', 'eeg_mmpe_s16_d6_r5', 'eeg_mmpe_s16_d6_r3', 'eeg_mmpe_s18_d6_r1', 'eeg_mmpe_s14_d6_r5', 'eeg_mmpe_s15_d6_r3', 'eeg_mmpe_s13_d6_r3', 'eeg_mmpe_s13_d6_r2', 'eeg_mmpe_s20_d6_r5', 'eeg_mmpe_s17_d6_r5', 'eeg_mmpe_s12_d6_r2', 'eeg_mmpe_s19_d6_r5','eeg_mmpe_s19_d6_r2', 'eeg_mmpe_s16_d6_r1', 'eeg_mmpe_s20_d6_r3', 'eeg_mmpe_s17_d6_r4', 'eeg_mmpe_s19_d6_r4', 'eeg_mmpe_s18_d6_r4', 'eeg_mmpe_s19_d6_r1', 'eeg_mmpe_s17_d6_r1', 'eeg_mmpe_s11_d6_r2', 'eeg_mmpe_s16_d6_r4', 'eeg_mmpe_s19_d6_r3', 'eeg_mmpe_s15_d6_r4', 'eeg_mmpe_s12_d6_r3', 'eeg_mmpe_s13_d6_r5', 'eeg_mmpe_s14_d6_r4', 'eeg_mmpe_s15_d6_r1', 'eeg_mmpe_s20_d2_r2', 'eeg_mmpe_s17_d5_r3', 'eeg_mmpe_s14_d6_r1', 'eeg_mmpe_s10_d6_r2', 'eeg_mmpe_s13_d6_r4', 'eeg_mmpe_s11_d6_r3', 'eeg_mmpe_s18_d5_r3', 'eeg_mmpe_s16_d5_r5', 'eeg_mmpe_s18_d3_r2', 'eeg_mmpe_s20_d5_r1', 'eeg_mmpe_s18_d5_r2', 'eeg_mmpe_s18_d2_r2', 'eeg_mmpe_s18_d5_r5', 'eeg_mmpe_s12_d6_r5', 'eeg_mmpe_s17_d5_r2', 'eeg_mmpe_s11_d6_r5', 'eeg_mmpe_s18_d3_r3', 'eeg_mmpe_s12_d6_r4', 'eeg_mmpe_s9_d6_r2', 'eeg_mmpe_s16_d5_r3', 'eeg_mmpe_s15_d5_r3', 'eeg_mmpe_s16_d5_r2', 'eeg_mmpe_s18_d5_r1', 'eeg_mmpe_s15_d5_r5', 'eeg_mmpe_s20_d5_r4', 'eeg_mmpe_s11_d6_r4', 'eeg_mmpe_s17_d5_r5', 'eeg_mmpe_s13_d6_r1', 'eeg_mmpe_s19_d5_r4', 'eeg_mmpe_s18_d4_r3', 'eeg_mmpe_s15_d5_r2', 'eeg_mmpe_s14_d5_r2', 'eeg_mmpe_s8_d6_r2', 'eeg_mmpe_s19_d5_r1', 'eeg_mmpe_s20_d5_r2', 'eeg_mmpe_s12_d6_r1', 'eeg_mmpe_s10_d6_r3', 'eeg_mmpe_s18_d4_r2', 'eeg_mmpe_s16_d2_r4', 'eeg_mmpe_s14_d5_r3', 'eeg_mmpe_s16_d4_r5', 'eeg_mmpe_s17_d3_r3', 'eeg_mmpe_s10_d6_r4', 'eeg_mmpe_s17_d4_r3', 'eeg_mmpe_s14_d5_r5', 'eeg_mmpe_s13_d2_r4', 'eeg_mmpe_s17_d5_r1', 'eeg_mmpe_s7_d6_r2', 'eeg_mmpe_s19_d5_r3', 'ecg_rcmpe_s2_d6', 'ecg_rcmpe_s3_d6', 'ecg_rcmpe_s1_d6', 'ecg_rcmpe_s2_d5', 'ecg_rcmpe_s1_d5', 'ecg_rcmpe_s3_d5', 'ecg_rcmpe_s2_d4', 'ecg_rcmpe_s3_d4']
