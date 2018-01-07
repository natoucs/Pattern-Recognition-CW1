import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from scipy.misc import imshow
from scipy.misc import toimage
from sklearn import metrics
import time
import math
import itertools

face=sio.loadmat("face.mat")

plt.plot(range(17))

arrayOfImages= np.array(face['X'])

training=[] #70% of the set
testing=[] #30% of the set

for i in range(52):
    for j in range(10*i,10*(i+1)): 
        if ((j-10*i)>6):
            testing.append(arrayOfImages[:,j])
        else:
            training.append(arrayOfImages[:,j])

training=np.array(training).T
testing=np.array(testing).T

#Help for data visualization

listOfImages=[None]*520
classOfImages={}

# for j in range(520):
#     listOfImages[j]=np.rot90(np.resize(arrayOfImages[:,j],(46,56)),-1)


# for i in range(52):
#     # print("i="+str(i))
#     tpp=[]
#     classOfImages[i]=tpp
#     for j in range(10*i,10*(i+1)):    
#         classOfImages[i].append(listOfImages[j])

average_image=np.mean(training,axis=1)

Atmp=[None]*364
for i in range(364):
    Atmp[i]=training[:,i]-average_image

A=np.array(Atmp).T

# start1=time.time()

S1=(1/364)*A.dot(A.T)
eigenval1,eigenvec1=np.linalg.eigh(S1)
idx1=eigenval1.argsort()[::-1]

eigenval1=eigenval1[idx1]
eigenvec1=eigenvec1[:,idx1]

# end1=time.time()

# print("Q1 done in "+str(end1-start1)+ " seconds")

start2=time.time()

S2=(1/364)*(A.T).dot(A)

eigenval2,eigenvec2=np.linalg.eigh(S2)
idx2=eigenval2.argsort()[::-1]


eigenval2=eigenval2[idx2]
eigenvec2=eigenvec2[:,idx2]

end2=time.time()

print("Q2 done in "+str(end2-start2)+ " seconds")

check=A.dot(eigenvec2)

#reconstructed u (=Av normalized)
check_norm=check/np.linalg.norm(check,axis=0)

print('Sum of Eigenvalues is '+str(np.sum(eigenval2)))

# print("step1")
# #Differences of dimension: 520x520 for eigenval2 and 2576x2576 for eigenval1; first 520 of eigenval1 are the same as eigenval2 within a tolerance of 1.29e-9 
# # print(np.allclose(eigenval1[:364],eigenval2,rtol=0,atol=1.29e-9))
# #Main difference is compitation time; ~0.2s vs ~16s
# # print(np.allclose(np.absolute(check_norm[:,:363]),np.absolute(eigenvec1[:,:363]))) #= True Vectors are the same with some eigenvectors multiplied by -1 but span the same space so still verify u=Av as -v is also an eigenvector

Atest=testing-np.array([average_image]*156).T

J=[]
for i in range(364):
    m=(A.T).dot(check_norm[:,:i])
    reconstructed= np.array([average_image]*364).T  +check_norm[:,:i].dot(m.T)
    diff=np.absolute(training-reconstructed)
    acc=0
    for j in range(364):
        acc+=diff[:,j].dot(diff[:,j].T)
    
    J.append(acc/364)


Jtest=[]
for i in range(364):
    m=(Atest.T).dot(check_norm[:,:i])
    reconstructed= np.array([average_image]*156).T  +check_norm[:,:i].dot(m.T)
    diff=np.absolute(testing-reconstructed)
    acc=0
    for j in range(156):
        acc+=diff[:,j].dot(diff[:,j].T)
    
    Jtest.append(acc/156)

print("step2")
plt.figure(1)
#Distortion error blue is Train green is Test
plt.plot(range(364),J,'b',range(364),Jtest,'g')
 
plt.figure(2)

# #eventually calculate MSE for each of the 3 images to comment the quality of image

# ################## M=50 #################
# m50=(A.T).dot(check_norm[:,:50])
# reconstructed50= np.array([average_image]*364).T  +check_norm[:,:50].dot(m50.T)


# m50test=(Atest.T).dot(check_norm[:,:50])
# reconstructed50test= np.array([average_image]*156).T  +check_norm[:,:50].dot(m50test.T)

# #image0
# plt.imshow(np.rot90(np.resize(reconstructed50[:,0],(46,56)),-1),cmap='gray')
 
# plt.figure(3)

# #image1
# plt.imshow(np.rot90(np.resize(reconstructed50[:,40],(46,56)),-1),cmap='gray')
 
# plt.figure(4)

# #test0
# plt.imshow(np.rot90(np.resize(reconstructed50test[:,110],(46,56)),-1),cmap='gray')
 
# plt.figure(5)


# ################## M=100 #################
# m100=(A.T).dot(check_norm[:,:100])
# reconstructed100= np.array([average_image]*364).T  +check_norm[:,:100].dot(m100.T)


# m100test=(Atest.T).dot(check_norm[:,:100])
# reconstructed100test= np.array([average_image]*156).T  +check_norm[:,:100].dot(m100test.T)

# #image0
# plt.imshow(np.rot90(np.resize(reconstructed100[:,0],(46,56)),-1),cmap='gray')
 
# plt.figure(6)

# #image1
# plt.imshow(np.rot90(np.resize(reconstructed100[:,40],(46,56)),-1),cmap='gray')
 
# plt.figure(7)

# #test0
# plt.imshow(np.rot90(np.resize(reconstructed100test[:,110],(46,56)),-1),cmap='gray')
 
# plt.figure(8)

# ################## M=200 #################
# m200=(A.T).dot(check_norm[:,:200])
# reconstructed200= np.array([average_image]*364).T  +check_norm[:,:200].dot(m200.T)


# m200test=(Atest.T).dot(check_norm[:,:200])
# reconstructed200test= np.array([average_image]*156).T  +check_norm[:,:200].dot(m200test.T)

# #image0
# plt.imshow(np.rot90(np.resize(reconstructed200[:,0],(46,56)),-1),cmap='gray')
 
# plt.figure(9)

# #image1
# plt.imshow(np.rot90(np.resize(reconstructed200[:,40],(46,56)),-1),cmap='gray')
 
# plt.figure(10)

# #test0
# plt.imshow(np.rot90(np.resize(reconstructed200test[:,110],(46,56)),-1),cmap='gray')
 
# plt.figure(11)



# #IMAGE0 Ref
# plt.imshow(np.rot90(np.resize(training[:,0],(46,56)),-1),cmap='gray')
 
# plt.figure(12)


# #IMAGE1 Ref
# plt.imshow(np.rot90(np.resize(training[:,40],(46,56)),-1),cmap='gray')
 
# plt.figure(13)

# #TEST0 Ref
# plt.imshow(np.rot90(np.resize(testing[:,110],(46,56)),-1),cmap='gray')
 
# plt.figure(14)


# plt.plot(idx1[:364],eigenval2[:364],'b^')
 
# plt.figure(15)



print("step3")

#NN

prediction=[]
prediction_train=[]

for j in range(364):
    min_error=[]
    print(j)
    for i in range(156):
        m=(Atest[:,i].T).dot(check_norm[:,:j])
        reconstructed= np.array([average_image.T  +check_norm[:,:j].dot(m.T)]*364).T
        min_error.append(np.argmin(np.linalg.norm(np.absolute(training-reconstructed),axis=0)))
        
        

    error_min_ref= [math.ceil(x/7) for x in min_error]
    reference=[math.ceil((x)/3) for x in range(1,157)]

    good_prediction=0
    for i in range(156):
        if (error_min_ref[i]==reference[i]):
            good_prediction+=1

    prediction.append(100*good_prediction/156)


plt.plot(range(364),prediction,'b')

plt.show()

j=364
start3=time.time()
min_error=[]
# print(j)
for i in range(156):
    m=(Atest[:,i].T).dot(check_norm[:,:j])
    reconstructed= np.array([average_image.T  +check_norm[:,:j].dot(m.T)]*364).T
    min_error.append(np.argmin(np.linalg.norm(np.absolute(training-reconstructed),axis=0)))
    

end3=time.time()
print("M=364: "+str(end3-start3))

error_min_ref= [math.ceil((x)/7) for x in min_error]
reference=[math.ceil((x)/3) for x in range(1,157)]
print(min_error)
print(error_min_ref)

# print("class 14 "+str(error_min_ref[39])+ " "+ str(error_min_ref[40])+" "+str(error_min_ref[41]))
# print("class 30 "+str(error_min_ref[87])+ " "+ str(error_min_ref[88])+" "+str(error_min_ref[89]))

# print("ref class 14 "+str(reference[39])+ " "+ str(reference[40])+" "+str(reference[41]))
# print("ref class 30 "+str(reference[87])+ " "+ str(reference[88])+" "+str(reference[89]))

plt.figure()
#example wrong
plt.imshow(np.rot90(np.resize(training[:,69],(46,56)),-1),cmap='gray')
plt.figure()
plt.imshow(np.rot90(np.resize(testing[:,1],(46,56)),-1),cmap='gray')
plt.figure()
#example correct
plt.imshow(np.rot90(np.resize(testing[:,92],(46,56)),-1),cmap='gray')
plt.figure()
plt.imshow(np.rot90(np.resize(training[:,212],(46,56)),-1),cmap='gray')
plt.figure()
#example partial
plt.imshow(np.rot90(np.resize(testing[:,42],(46,56)),-1),cmap='gray')
plt.figure()
#correct
plt.imshow(np.rot90(np.resize(training[:,104],(46,56)),-1),cmap='gray')
plt.figure()

#example partial
plt.imshow(np.rot90(np.resize(testing[:,43],(46,56)),-1),cmap='gray')
plt.figure()
#incorrect
plt.imshow(np.rot90(np.resize(training[:,223],(46,56)),-1),cmap='gray')
# plt.figure()
plt.show()
 



conf_matrix=metrics.confusion_matrix(reference,error_min_ref)
conf_matrix=np.array(conf_matrix)
summary={0:[], 1:[],2:[],3:[]}
for i in range(52):
    if(conf_matrix[i,i]==3):
        summary[3].append(i)
    if(conf_matrix[i,i]==2):
        summary[2].append(i)
    if(conf_matrix[i,i]==1):
        summary[1].append(i)
    if(conf_matrix[i,i]==0):
        summary[0].append(i)

print(summary)

good_prediction=0
for i in range(156):
    if (error_min_ref[i]==reference[i]):
        # print(i)
        good_prediction+=1

print(100*good_prediction/156)


# j=50
# start3=time.time()
# min_error=[]
# # print(j)
# for i in range(156):
#     m=(Atest[:,i].T).dot(check_norm[:,:j])
#     reconstructed= np.array([average_image.T  +check_norm[:,:j].dot(m.T)]*364).T
#     min_error.append(np.argmin(np.linalg.norm(np.absolute(training-reconstructed),axis=0)))

# end3=time.time()  
# print("M=50: "+str(end3-start3))  

# error_min_ref= [math.ceil(x/7) for x in min_error]
# reference=[math.ceil((x)/3) for x in range(1,157)]

# conf_matrix=metrics.confusion_matrix(reference,error_min_ref)
# conf_matrix=np.array(conf_matrix)
# summary={0:[], 1:[],2:[],3:[]}
# for i in range(52):
#     if(conf_matrix[i,i]==3):
#         summary[3].append(i)
#     if(conf_matrix[i,i]==2):
#         summary[2].append(i)
#     if(conf_matrix[i,i]==1):
#         summary[1].append(i)
#     if(conf_matrix[i,i]==0):
#         summary[0].append(i)

# print(summary)

# good_prediction=0
# for i in range(156):
#     if (error_min_ref[i]==reference[i]):
#         good_prediction+=1

# print(100*good_prediction/156)



# print(conf_matrix)

