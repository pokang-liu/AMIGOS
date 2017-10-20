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
	return Fisher

	#sort the fisher from small to large

	feature_idx=np.arange(212)
	Fisher_sorted = np.array(Fisher).argsort()
	#####################################################
	sorted_feature_idx = feature_idx[Fisher_sorted[::-1]]
	#####################################################

#########add this at line 82!!!!######################
train_a_labels=np.array(train_a_labels)
train_a_labels0 = np.where(train_a_labels>0) 
train_a_labels1 = np.where(train_a_labels<1) 
###########
train_a_labels0=np.array(train_a_labels0)
############
Fisher=[]
train_data0 = np.delete(train_data, train_a_labels0, axis=0)
train_data1= np.delete(train_data, train_a_labels1, axis=0)

mean_train_data0=np.mean(train_data0,axis=0)
mean_train_data1=np.mean(train_data1,axis=0)

std_train_data0=np.std(train_data0)
std_train_data1=np.std(train_data1)
std_sum=(std_train_data1)*(std_train_data1)+std_train_data0*std_train_data0


Fisher=(mean_train_data0-mean_train_data1)/std_sum
Fisher=np.array(Fisher)
print(Fisher.shape)

#sort the fisher from small to large

feature_idx=np.arange(212)
Fisher_sorted = np.array(Fisher).argsort()
#####################################################
sorted_feature_idx = feature_idx[Fisher_sorted[::-1]]
#####################################################
