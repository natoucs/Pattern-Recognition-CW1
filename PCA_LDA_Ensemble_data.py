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
T = 3 #only modify this one.
L = 364 #share data between T training replicates
m_pca = 364-1 
m_lda = 52-1
p = 2576

# Perform PCA to get Wpca, assume w_pca is np.array, w_pca p*m_pca
pca_avg = np.mean(training,axis=1)  #pca_avg p*1, use face instead of x to not mix anything
pca_sub = training - np.array([pca_avg]*364).T #pca_sub p*364
pca_cov = ( pca_sub . dot( pca_sub.T ) ) / 364.0 #pca_cov p*p, is symmetric so eigenvalues are real
pca_eigenvalue, pca_eigenvector = np.linalg.eigh(pca_cov) # pca_eigval p*1 , pca_eigvec p*p, eigh prevents complex values
pca_index = pca_eigenvalue.argsort()[::-1] 
pca_eigenvalue = pca_eigenvalue[pca_index] #sort with largest in position 0
pca_eigenvector = pca_eigenvector[:,pca_index] #eigenvectors are in columns
w_pca = pca_eigenvector[:,:m_pca] #w_pca: p*m_pca

# Put face image input on PCA dimensional space using w_pca & do it also for testing btw
training_pca = w_pca.T . dot(pca_sub) #training_pca: m_pca*364, pca_sub: 2576*364
pca_sub_t = testing - np.array([pca_avg]*156).T #compute (image-mean) for testing set
testing_pca = w_pca.T . dot(pca_sub_t) #testing_pca: m_pca*156

print "1"
#Generate T training sets in master_training m_pca*L*T
master_training = np.zeros((m_pca, L, T))
class_reference = np.zeros((T,L)) #each row is L class references from replicate T
class_number = np.zeros((T,52)) 
for i in range(T):
    index = np.random.randint(0,364,L) #returns L index between 0 and 363
    master_training[:,:,i] = training_pca[:,index]
    class_reference[i,:] = index/7 #keep track of reference class of this training set
    for j in range(52): #may not be necessary
        class_number[i,j] = np.sum(class_reference[i,:] == j) #count number of image of each class and store it

print "2"
#Construct an LDA classifier from each T sets
master_wlda = np.zeros((m_pca, m_lda, T))
for k in range(T):
    training =  master_training[:,:,k] #training m_pca*L
    # compute Sw & Sb
    m_i = np.zeros((m_pca, 52))
    for i in range(0, 52*7, 7): #0..7..14..51*7
        m_i[:,i/7] = np.mean(training[:,class_reference[k,:]==i/7],axis=1) #m_i 2576*52
    m = np.mean(m_i,axis=1) #m is 2576*1
    m_sub = m_i - np.array([m]*52).T # m_sub 2576*52, m_sub at col 0 is m1-m, [m] transforms to list so need reconversion to np.array
    Sb = np.zeros((2576, 2576)) # m_sub[:,i]*m_sub[:,i].T = 2576*1 x 1*2576
    for i in range(52):
        Sb += m_sub[:,i].dot(m_sub[:,i]).T #(m1-m).(m1-m).T ... 52 times ... acc in Sb   
    Sw = np.zeros((2576, 2576))
    for i in range(0, 52*7, 7):
        x_sub = training[:,i:i+7] - np.array([m_i[:,i/7]]*7).T #x_sub 2576*7, x_sub at col 0 is x1-m1
        for j in range (7):
            Sw += x_sub[:,j].dot(x_sub[:,j]).T #(x1-m1).(x1-m1).T ... 7 times for m1 ... for 52 classes ... acc in Sw

    # Use Wpca on Sb,Sw,w_pca to get eigenvectors with first largest M_lda eigenvalues  
    tmp_b = (w_pca.T) . dot(Sb) . dot(w_pca) # tmp_b m_pca*m_pca
    tmp_w = (w_pca.T) . dot(Sw) . dot(w_pca) # tmp_a m_pca*m_pca
    tmp_f = np.linalg.inv( tmp_w ) . dot( tmp_b ) #tmp_f m_pca*m_pca
    lda_eigenvalue, lda_eigenvector = np.linalg.eigh(tmp_f) 
    lda_index = lda_eigenvalue.argsort()[::-1] 
    lda_eigenvalue = lda_eigenvalue[lda_index] 
    lda_eigenvector = lda_eigenvector[:,lda_index] 
    master_wlda[:,:,k] = lda_eigenvector[:,:m_lda] #master_wlda: m_pca*m_lda*T

print "3"
#Compute the transform of T LDA classifier using master_wlda m_pca*m_lda*T
y_train = np.zeros((m_lda,L,T))
y_test = np.zeros((m_lda,156))
for i in range(T): #master_wlda: m_pca*m_lda*T
    y_train[:,:,i] = master_wlda[:,:,i].T . dot(master_training[:,:,i]) # y_train: m_lda*L*T
    y_test = master_wlda[:,:,i].T . dot(testing_pca) #y_test: m_lda*156*T

#Nearest neighbour classifier
y_result = np.zeros((m_lda,L,T))
class_prediction = np.zeros((T+2,156),np.int8) 
for j in range(T):
    for i in range (156): #for each element in the test set
        y_result[:,:,j] = y_train[:,:,j] - np.array( [ y_test[:,i] ] * L ).T #y_result: m_lda*L*T, try each element in the training set
        index_result = np.argmin(np.linalg.norm(y_result[:,:,j],axis=0)) 
        class_prediction[j,i] = class_reference[j, index_result] #returns number of class predicted by classifier L

#Fusion rule : Majority voting
true_prediction = 0
for i in range (156):
    counts = np.bincount(class_prediction[:T,i]) 
    class_prediction[T,i] = np.argmax(counts)  #loads before last row with fusioned output of classifiers
    class_prediction[T+1,i] = (i / 3)+1 #loads in the last row the class_reference
    if (class_prediction[T+1,i] == class_prediction[T,i]): #if class value is the same
        true_prediction += 1

#Results 
print("(%i,%i,%i,%i)->(%i)" % (T,L,m_pca,m_lda,true_prediction) )  
np.set_printoptions(threshold='nan', linewidth=210)
print class_prediction[:,:50]
print class_prediction[:,50:100]
print class_prediction[:,100:156]

