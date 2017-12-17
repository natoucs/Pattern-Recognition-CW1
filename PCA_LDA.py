import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from scipy.misc import imshow
from scipy.misc import toimage
import time
import math

face = sio.loadmat("face.mat")
arrayOfImages = np.array(face['X']) #passing to numpy array
training = [] #70% of the set
testing = [] #30% of the set

for i in range(52): #52 classes of 10 images each - 0..519 for 520 images
    for j in range(10*i,10*(i+1)): 
        if ((j-10*i)>6): #put in testing 3 out of 10
            testing.append(arrayOfImages[:,j]) #156*2576(=46*56=nbr pixels)
        else: #put in training 7 out of 10
            training.append(arrayOfImages[:,j]) #364*2576(=46*56=nbr pixels)

training = np.array(training).T #now: (2576, 364)
testing = np.array(testing).T #now: (2576, 156)
#NOW one col is a full image

# LDA-PCA-NN
# Define c=52, N=10/class, D=2576=length of one image

print ("1")
# compute matrix m_i & m
m_i = np.zeros((2576, 52))
for i in range(0, 52*7, 7): #0..7..14..51*7
    m_i[:,i/7] = np.mean(training[:,i:i+7],axis=1) #m_i 2576*52
m = np.mean(m_i,axis=1) #m is 2576*1

# compute matrix mi-m & Sb
m_sub = m_i - np.array([m]*52).T # m_sub 2576*52, m_sub at col 0 is m1-m, [m] transforms to list so need reconversion to np.array
Sb = np.zeros((2576, 2576)) # m_sub[:,i]*m_sub[:,i].T = 2576*1 x 1*2576
for i in range(52):
    Sb += m_sub[:,i].dot(m_sub[:,i]).T #(m1-m).(m1-m).T ... 52 times ... acc in Sb
Sb = np.multiply(7.0,Sb)    

# compute matrix x-mi & Sw 
Sw = np.zeros((2576, 2576))
for i in range(0, 52*7, 7):
     x_sub = training[:,i:i+7] - np.array([m_i[:,i/7]]*7).T #x_sub 2576*7, x_sub at col 0 is x1-m1
     for j in range (7):
        Sw += x_sub[:,j].dot(x_sub[:,j]).T #(x1-m1).(x1-m1).T ... 7 times for m1 ... for 52 classes ... acc in Sw

print ("2")
# Perform PCA to get Wpca, assume w_pca is np.array, W_pca 2576*M_pca
pca_avg = np.mean(training,axis=1)  #pca_avg 2576*1, use face instead of x to not mix anything
pca_sub = training - np.array([pca_avg]*364).T #pca_sub 2576*364
pca_cov = ( pca_sub . dot( pca_sub.T ) ) / 364.0 #pca_cov 2576*2576, is symmetric so eigenvalues are real
pca_eigenvalue, pca_eigenvector = np.linalg.eigh(pca_cov) # pca_eigval 2576*1 , pca_eigvec 2576*2576, eigh prevents complex values
pca_index = pca_eigenvalue.argsort()[::-1] 
pca_eigenvalue = pca_eigenvalue[pca_index] #sort with largest in position 0
pca_eigenvector = pca_eigenvector[:,pca_index] #eigenvectors are in columns

#Choose k PCs containing >95% of the variance.
# total = np.sum(pca_eigenvalue)
# PC_percentage = 0
# m_pca = 0
# while (PC_percentage < 0.999):
#     PC_percentage += pca_eigenvalue[m_pca] / total
#     m_pca += 1  #takes until the p-1 index but okay since p++ after PC_percentage > 0.95 is computed

m_pca = 363 #364-1, keep all non-zero eigenvectors
w_pca = pca_eigenvector[:,:m_pca] #w_pca: 2576*m_pca

# Decide on m_pca and m_lda -> m_lda <= m_pca
# Can add Fisher_score to choose m_lda values 
m_lda = 51 #52-1

#Record results (m_pca,m_lda,good_proj)
# (1000,1,9) (1000,5,18) (1000,25,47) (1000,50,66) (1000,100,79) (1000,150,78) (1000,200,79) (1000,500,84) (1000,1000,86)
# (500,1,11) (500,5,12) (500,25,29) (500,50,36) (500,100,65) (500,150,69) (500,200,71) (500,500,86)
# (150,1,11) (150,5,15)(150,25,35) (150,50,58) (150,100,79) (150,150,84) (150,200,84)
# (50,1,8) (50,5,20) (50,25,67) (50,50,79) (50,100,79) (50,150,79) (50,200,79)
# (5,1,5) (5,5,42) (5,25,42) (5,50,42) (5,100,42) (5,150,42) (5,200,42)
# (3,1,6) (3,5,23) (3,25,23) (3,50,23) (3,100,23) (3,150,23)
# (363,51,71)

print ("3")
# Use Wpca on Sb,Sw,w_pca to get eigenvectors with first largest M_lda eigenvalues  
tmp_b = (w_pca.T) . dot(Sb) . dot(w_pca) # tmp_b m_pca*m_pca
tmp_w = (w_pca.T) . dot(Sw) . dot(w_pca) # tmp_a m_pca*m_pca
tmp_f = np.linalg.inv( tmp_w ) . dot( tmp_b ) #tmp_f m_pca*m_pca
lda_eigenvalue, lda_eigenvector = np.linalg.eigh(tmp_f) 
lda_index = lda_eigenvalue.argsort()[::-1] 
lda_eigenvalue = lda_eigenvalue[lda_index] 
lda_eigenvector = lda_eigenvector[:,lda_index] 
w_lda = lda_eigenvector[:,:m_lda] #w_lda: m_pca*m_lda

# Put face image input on PCA dimensional space using w_pca
training_pca = w_pca.T . dot(pca_sub) #training_pca: m_pca*364, pca_sub: 2576*364
pca_sub_t = testing - np.array([pca_avg]*156).T #compute (image-mean) for testing set
testing_pca = w_pca.T . dot(pca_sub_t) #testing_pca: m_pca*156

# Launch on LDA dimensional space #w_lda: m_pca*m_lda
y_train = w_lda.T . dot(training_pca) # y_train: m_lda*364, each col = 1 sample 
y_test = w_lda.T . dot(testing_pca) #y_test: m_lda*156, each col = 1 sample 

print ("4")
#Nearest neighbour classifier
confusion_matrix = np.zeros((52,52),np.int8) #ref in colums, prediction in row
true_prediction = 0
for i in range (156): #for each element in the test set
    y_result = y_train - np.array( [ y_test[:,i] ] * 364 ).T #y_result: m_lda*364, try each element in the training set
    index_result = np.argmin(np.linalg.norm(y_result,axis=0)) #return index of smallest normalised column of y_result 
    class_prediction = index_result / 7 #says which class test is predicted to be
    class_reference = i / 3  #says which class test is truly from
    confusion_matrix[class_prediction, class_reference] += 1 #for the confusion matrix only
    if (class_prediction == class_reference): #if class value is the same
        true_prediction += 1
        successed_test = i  # successfully classified image to show
        successed_prediction = index_result #show image of the same class
    else:
         failed_test = i #failed classified image to show
         failed_prediction = index_result  #show image who looked similar enough to do an error
         print "(fail_pred,true)->(%i,%i)" %(class_prediction+1, class_reference+1)
    
#plot the LDA success and failure case + confusion matrix  
print("(%i,%i,%i)" % (m_pca,m_lda,true_prediction) )
np.set_printoptions(threshold='nan', linewidth=210)
print confusion_matrix
plt.plot(range(4)) #tell xx windows are gg to be used
plt.figure(1)
plt.title('successed_classification')
plt.imshow(np.rot90(np.resize(testing[:,successed_test],(46,56)),-1),cmap='gray')
plt.figure(2)
plt.title('successed_similar')
plt.imshow(np.rot90(np.resize(training[:,successed_prediction],(46,56)),-1),cmap='gray')
plt.figure(3)
plt.title('failed_classification')
plt.imshow(np.rot90(np.resize(testing[:,failed_test],(46,56)),-1),cmap='gray')
plt.figure(4)
plt.title('failed_similar')
plt.imshow(np.rot90(np.resize(training[:,failed_prediction],(46,56)),-1),cmap='gray')
plt.show()

summary={0:[], 1:[],2:[],3:[]}
for i in range(52):
    if(confusion_matrix[i,i]==3):
        summary[3].append(i)
    if(confusion_matrix[i,i]==2):
        summary[2].append(i)
    if(confusion_matrix[i,i]==1):
        summary[1].append(i)
    if(confusion_matrix[i,i]==0):
        summary[0].append(i)
print summary