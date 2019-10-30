import pandas as pd
import numpy as np 
import operator
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#
# Loading Data

columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 
           'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
           'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
           'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
           'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl',
           'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 
           'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
           'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
           'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[',
           'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'is_spam']

spambase = pd.read_csv("spambase.data",  delimiter=",",names=columns, header=None)



#Create 100 record version of training data with same fraction of SPAM emails as the original data

spambase_spam = spambase[spambase['is_spam']==1].sample(n=78)
spambase_ham = spambase[spambase['is_spam']==0].sample(n=122)
kNN_200 = pd.concat([spambase_spam, spambase_ham], ignore_index=True).reset_index()

kNN_200 = kNN_200.drop(['index'],axis=1)

kNN_200 = kNN_200.sample(frac=1)


knn_features = kNN_200.drop(['is_spam'],axis=1)
knn_response = kNN_200['is_spam']

xTrain_knn, xTest_knn, yTrain_knn, yTest_knn = train_test_split(knn_features,knn_response, test_size = 0.5, random_state = 123, stratify=knn_response)

std_scaler = preprocessing.StandardScaler()
xTrain_knn= std_scaler.fit_transform(xTrain_knn.values)
xTest_knn= std_scaler.fit_transform(xTest_knn.values)



class kNN():


    def __init__(self, feature_training_set, feature_test_set, response_training_set, response_test_set, k):  
        self.xTrain = feature_training_set
        self.xTest = feature_test_set
        self.yTrain = response_training_set
        self.yTest = response_test_set
        self.k = k

    def __computeEuclideanDist(self,x,y):
        '''
        Function to compute n-dimensional Euclidean Distance
        inputs x , y must be numpy arrays
        '''
        return np.sqrt(np.sum(np.square(x-y),axis=1))
    
            
    def __getvotes(self, topk_labels):
        votes = {}
        for l in topk_labels:
            if l not in votes:
                votes[l] = 1
            else:
                votes[l] += 1        
        return votes
        
    
    
    def Fit_Predict(self):
    
        '''
        Calculates distance metrics for all points in testing and classifes 
        the test point by majority voting 
    
        '''
        
        xTrain = self.xTrain
        xTest  = self.xTest
        yTrain = self.yTrain
        yTest  = self.yTest 
        k      = self.k
        
        # asserting k can't be larger than number of samples.
        assert k <= len(xTest),  

        n = xTest.shape[0]
        ntrain = xTrain.shape[0]
        yTrain = np.array(yTrain).reshape((yTrain.shape[0],1))
        
        predictions =[]
        
        #compute euclidean distances 
        for i in range(n):    
            euclid_dist = self.__computeEuclideanDist(xTrain, xTest[i,:])
            
            euclid_dist = euclid_dist.reshape((ntrain,1))
            
            #concatenating distances to xTrain
            full_set = np.concatenate((xTrain,yTrain),axis=1)
    
            data_withdist = np.concatenate((full_set,euclid_dist),axis=1)

            #sort array in ascending order (i.e., so that smallest distances come first)
            #but we need to sort by the last column, 
            #whilst keeping everything in the same row tied together (argsort)
            sorted_array = data_withdist[data_withdist[:,-1].argsort()]
                        
            # slicling top-k labels from sorted array
            topk_labels= sorted_array[range(k),-2]

            votes = self.__getvotes(topk_labels)
  
            #Predicting label by finding the most voted class
            pred = max(votes.items(), key=operator.itemgetter(1))
            pred_return = pred[0]
            
            predictions.append(pred_return)
            
            predictions = np.asarray(predictions)
            
        return predictions
        


kvals=[2,3,5,9]

for k in kvals:
    MODEL = kNN(xTrain_knn,xTest_knn,yTrain_knn,k)
    preds = MODEL.Fit_Predict()
    print("\n++++++++++++++++++ k = {} ++++++++++++++++++\n".format(k))
    print("Accuracy Score:",accuracy_score((yTest_knn), preds))
    print("Error:",1 - accuracy_score((yTest_knn), preds))
