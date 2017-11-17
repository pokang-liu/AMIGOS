import xlsxwriter
import numpy as np
feat= np.load('a_rfe_feature_history.npy')


workbook = xlsxwriter.Workbook('a_rfe_feat_idx.xlsx')
worksheet = workbook.add_worksheet()
for i in range(feat.shape[0]):
    for j in range(feat.shape[1]):
        worksheet.write(i,j,feat[i][j])
workbook.close()


