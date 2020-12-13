from matplotlib import pyplot
import numpy as np
import pandas as pd
#------------------------------part 1--------------------------------
data = np.loadtxt(r'.\uspsdata\uspsdata.txt')
label = np.loadtxt(r'.\uspsdata\uspscl.txt')
#------------------------------part 2--------------------------------
for i in range(4):
    image = data[i].reshape(16,16)
    pyplot.imshow(image)
    pyplot.show()
#------------------------------part 3--------------------------------
def split(dataset,label):
    label = label.reshape(-1,1)
    temp = np.hstack((dataset,label))
    np.random.shuffle(temp)

    dataset = np.hsplit(temp, [256, 257])[0]
    label = np.hsplit(temp, [256, 257])[1]

    x_train = dataset[:int(0.6*len(dataset)),:]
    x_validation = dataset[int(0.6 * len(dataset)):int(0.8 * len(dataset)), :]
    x_test = dataset[int(0.8 * len(dataset)):, :]
    y_train = label[:int(0.6*len(label))]
    y_validation = label[int(0.6 * len(label)):int(0.8 * len(label))]
    y_test = label[int(0.8 * len(label)):]
    return x_train,y_train,x_validation,y_validation,x_test,y_test

x_train,y_train,x_validation,y_validation,x_test,y_test = split(data,label)
#------------------------------part 4--------------------------------
def k_distance(sample,x_train,y_train,k):
    distance= pd.DataFrame(columns=['dis','y'])
    for i in range(0,len(x_train)):
        d = np.linalg.norm(sample - x_train[i])
        dic={'dis':d,'y':y_train[i]}
        distance = distance.append(dic, ignore_index=True)
    distance = distance.sort_values(by='dis')
    distance = distance.reset_index()
    predict = 0
    for i in range(k):
        predict += distance.at[i,'y']
    if predict >= 0:
        return +1
    else:
        return -1
def KNN(x_train,y_train,x_test,y_test,k):
    sum,flag =0,0
    for i in range(0,len(x_test)):
        sum +=1
        if k_distance(x_test[i],x_train,y_train,k) == y_test[i]:
            flag +=1
        else:
            image = x_test[i].reshape(16,16)
            pyplot.imshow(image)
            pyplot.show()

    print("a ",k,"-nearest neighbor classifier using the training data and predict the labels of the images in the testing data ==",flag/sum)
    print("error = ",1-flag/sum)

KNN(x_train,y_train,x_test,y_test,1)
#------------------------------part 5--------------------------------
def KNN_2(x_train,y_train,x_test,y_test,k):
    sum,flag =0,0
    for i in range(0,len(x_test)):
        sum +=1
        if k_distance(x_test[i],x_train,y_train,k) == y_test[i]:
            flag +=1
    print("error with k=",k," = ",1-flag/sum)


KNN_2(x_train,y_train,x_test,y_test,1)
KNN_2(x_train,y_train,x_test,y_test,3)
KNN_2(x_train,y_train,x_test,y_test,5)
KNN_2(x_train,y_train,x_test,y_test,6)
KNN_2(x_train,y_train,x_test,y_test,7)
KNN_2(x_train,y_train,x_test,y_test,9)
KNN_2(x_train,y_train,x_test,y_test,11)
KNN_2(x_train,y_train,x_test,y_test,13)
#------------------------------***part 6***--------------------------------
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train,y_train)
print(neigh.score(x_test,y_test))
#--------------------------------------------------------------------------
