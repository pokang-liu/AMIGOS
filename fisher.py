def fisher(features,label):
	Fisher=[]
	label=np.array(label)
	labels0 = np.where(labels>0) 
	labels1 = np.where(labels<1) 
	labels0=np.array(labels0)
	features0 = np.delete(features, labels0, axis=0)
	features1= np.delete(features, labels1, axis=0)

	mean_features0=np.mean(features0,axis=0)
	mean_features1=np.mean(features1,axis=0)

	std_features0=np.std(features0)
	std_features1=np.std(features1)
	std_sum=(std_features1)*(std_features1)+std_features0*std_features0


	Fisher=(abs(mean_features0-mean_features1))/std_sum
	Fisher=np.array(Fisher)
	print(Fisher.shape)
	

	#sort the fisher from small to large

	feature_idx=np.arange(data.shape[1])
	Fisher_sorted = np.array(Fisher).argsort()
	#Fisher_sorted[0] has the smallest score
	#####################################################
	#now Fisher_sorted[::-1]]'s head is the index with the largest score!!
	sorted_feature_idx = feature_idx[Fisher_sorted[::-1]]
	return sorted_feature_idx
	#####################################################

def feature_selection(h)
	#only select h features
	h_features=features[sorted_feature_idx[:h]]
	return h_features
	
	

