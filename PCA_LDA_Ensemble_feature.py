import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from scipy.misc import imshow
from scipy.misc import toimage
import time
import math
import random

face = sio.loadmat("face.mat")
arrayOfImages = np.array(face['X']) #passing to numpy array

training = [] #70% of the set
testing = [] #30% of the set

for i in range(52): #52 classes of 10 images each - 0..519 for 520 images
    for j in range(10*i,10*(i+1)): 
        if ((j-10*i)>6): #put in testing 3 out of 10
            testing.append(arrayOfImages[:,j]) #156*p(=46*56=nbr pixels)
        else: #put in training 7 out of 10
            training.append(arrayOfImages[:,j]) #364*p(=46*56=nbr pixels)

training = np.array(training).T #now: (p, 364)
testing = np.array(testing).T #now: (p, 156)
#NOW one col is a full image

#----------------------------------

# LDA-PCA-NN Ensemble
# Define c=52, N=10/class, D=p=length of one image
# Define parameters k N0 N1 m_pca
k = 5 
N0 = 50
N1 = 50
m_pca = 363 
m_lda = 51
p = 2576

# Perform PCA to get Wpca, assume w_pca is np.array, W_pca p*m_pca
pca_avg = np.mean(training,axis=1)  #pca_avg p*1, use face instead of x to not mix anything
pca_sub = training - np.array([pca_avg]*364).T #pca_sub p*364
pca_cov = ( pca_sub . dot( pca_sub.T ) ) / 364.0 #pca_cov p*p, is symmetric so eigenvalues are real
pca_eigenvalue, pca_eigenvector = np.linalg.eigh(pca_cov) # pca_eigval p*1 , pca_eigvec p*p, eigh prevents complex values
pca_index = pca_eigenvalue.argsort()[::-1] 
pca_eigenvalue = pca_eigenvalue[pca_index] #sort with largest in position 0
pca_eigenvector = pca_eigenvector[:,pca_index] #eigenvectors are in columns
w_pca = pca_eigenvector[:,:m_pca] #w_pca: p*m_pca

print "1"
#Create k random subspaces -> eigenvectors of dim N0+N1: p*(N0+N1)*k 
master_wpca = np.zeros((p, N0+N1, k))
for i in range(k):
    index_N1 = random.sample(range(N0, m_pca), N1) #([low],]high[,size)
    tmp = np.concatenate((w_pca[:,:N0], w_pca[:,index_N1]), axis=1) #tmp p*(N0+N1)
    master_wpca[:,:,i] = tmp #master_wpca: p * N0+N1 * k

#Construct k LDA classifiers

print "2"
#Compute Sw & Sb
m_i = np.zeros((p, 52))
for i in range(0, 52*7, 7): #0..7..14..51*7
    m_i[:,i/7] = np.mean(training[:,i:i+7],axis=1) #m_i p*52
m = np.mean(m_i,axis=1) #m is p*1
m_sub = m_i - np.array([m]*52).T # m_sub p*52, m_sub at col 0 is m1-m, [m] transforms to list so need reconversion to np.array
Sb = np.zeros((p, p)) # m_sub[:,i]*m_sub[:,i].T = p*1 x 1*p
for i in range(52):
    Sb += m_sub[:,i].dot(m_sub[:,i]).T #(m1-m).(m1-m).T ... 52 times ... acc in Sb
np.multiply(7.0,Sb)    
Sw = np.zeros((p, p))
for i in range(0, 52*7, 7):
     x_sub = training[:,i:i+7] - np.array([m_i[:,i/7]]*7).T #x_sub p*7, x_sub at col 0 is x1-m1
     for j in range (7):
        Sw += x_sub[:,j].dot(x_sub[:,j]).T #(x1-m1).(x1-m1).T ... 7 times for m1 ... for 52 classes ... acc in Sw

print "3"
#Compute k LDA classifiers in master_wlda
master_wlda = np.zeros((N0+N1, m_lda, k))
for i in range(k):
    tmp_b = ( master_wpca[:,:,i].T) . dot(Sb) . dot( master_wpca[:,:,i]) 
    tmp_w = ( master_wpca[:,:,i].T) . dot(Sw) . dot( master_wpca[:,:,i]) 
    tmp_f = np.linalg.inv( tmp_w ) . dot( tmp_b ) #tmp_f (N0+N1)*(N0+N1)
    lda_eigenvalue, lda_eigenvector = np.linalg.eigh(tmp_f) 
    lda_index = lda_eigenvalue.argsort()[::-1] 
    lda_eigenvalue = lda_eigenvalue[lda_index] 
    lda_eigenvector = lda_eigenvector[:,lda_index] 
    master_wlda[:,:,i]  = lda_eigenvector[:,:m_lda] #master_wlda: N0+N1 * m_lda * k

print "4"
# Project input face data to each k subspace using master_wpca
training_pca = np.zeros((N0+N1,364,k))
testing_pca = np.zeros((N0+N1,156,k))
pca_sub_t = testing - np.array([pca_avg]*156).T #pca_sub_t: p*156
for i in range(k): #master_wpca: p * N0+N1 * k
    training_pca[:,:,i] = master_wpca[:,:,i].T . dot(pca_sub) #training_pca: (N0+N1)*364*k, pca_sub: p*364
    testing_pca[:,:,i] = master_wpca[:,:,i].T . dot(pca_sub_t) #testing_pca: (N0+N1)*156*k #pca_sub_t: p*156

print "5"
#Compute the transform of LDA classifier using master_wlda 
y_train = np.zeros((m_lda,364,k))
y_test = np.zeros((m_lda,156,k))
for i in range(k): #master_wlda: N0+N1 * m_lda * k
    y_train[:,:,i] = master_wlda[:,:,i].T . dot(training_pca[:,:,i]) # y_train: m_lda*364*k
    y_test[:,:,i] = master_wlda[:,:,i].T . dot(testing_pca[:,:,i]) #y_test: m_lda*156*k

print "6"
#Nearest neighbour classifier
y_result = np.zeros((m_lda,364,k))
class_prediction = np.zeros((k+2,156),np.int8) 
for j in range(k):
    for i in range (156): #for each element in the test set
        y_result[:,:,j] = y_train[:,:,j] - np.array( [ y_test[:,i,j] ] * 364 ).T #y_result: m_lda*364*k, try each element in the training set
        index_result = np.argmin(np.linalg.norm(y_result[:,:,j],axis=0)) 
        class_prediction[j,i] = (index_result / 7)+1 #returns number of class predicted by classifier k
       
#Fusion rule : Majority voting
true_prediction = 0
for i in range (156):
    counts = np.bincount(class_prediction[:k,i]) 
    class_prediction[k,i] = np.argmax(counts)  #loads before last row with fusioned output of classifiers
    class_prediction[k+1,i] = (i / 3)+1 #loads in the last row the class_reference
    if (class_prediction[k+1,i] == class_prediction[k,i]): #if class value is the same
        true_prediction += 1

#Results 
print("(%i,%i,%i,%i,%i)->(%i)" % (k,N0,N1,m_pca,m_lda,true_prediction) )  
np.set_printoptions(threshold='nan', linewidth=210)
print class_prediction[:,:50]
print class_prediction[:,50:100]
print class_prediction[:,100:156]
# (5,50,50,363,51)->(81) (5,50,100,363,51)->(78) (5,50,200,363,51)->(73)
# (5,10,50,363,51)->(70) 
# (3,50,150,363,51)->(72)
# (7,50,50,363,51)->(80) (7,50,25,363,51)->(75) (7,100,50,363,51)->(76)


