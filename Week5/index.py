from random import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier   
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import f1_score
import joblib


from tensorflow import keras

#Do em xai may macbook m1 nen khong the cai tensorflow de chay duoc tren jupyter notebook
#Nen em se chay trong moi truong ao tren terminal, mac du hoi bat tien, mong thay thong cam

#Dataset la FASHION_MNIST
#Dataset co tat ca 9 label

#Load data FASHION_MNIST
(X_train,y_train),(X_test,y_test)= keras.datasets.fashion_mnist.load_data()

# 1.2. Reshape to 2D array: each row has 784 features
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

#O day co mot dictionaries cua fashion_mnist
dictionaries_fashion_mnist={
    '0':'T-shirt/top',
    '1':'Trouser',
    '2':'Pullover',
    '3':'Dress',
    '4':'Coat',
    '5':'Sandal',
    '6':'Sandal',
    '7':'Sneaker',
    '8':'Bag',
    '9':'Ankle boot'
}        

# 1.3. Plot a digit image   
#Plot_digit la ham de ve hinh
def plot_digit(data, label = 'unspecified', showed=True):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("Digit: " + dictionaries_fashion_mnist[str(label)])
    #plt.axis("off")
    if showed:
        plt.show()
        
#id = 1 thi la tshirt/top
sample_id = 1
# plot_digit(X_train[sample_id], y_train[sample_id])

#Em se chia thanh 2 class, 1 class chi co T-shirt/top, class con lai khong co T-shirt/top
# Classify 2 classes (T-shirt/top or not T-shirt/top)
y_train_Tshirt = (y_train == 0)
y_test_Tshirt = (y_test == 0)

# print(y_train_Tshirt)

#Create model SGD_Classifier
sgd_clf = SGDClassifier(random_state = 42)

#Chung ta load SGD Classifier va du doan
new_run = False
if new_run:
    sgd_clf.fit(X_train,y_train_Tshirt)
    joblib.dump(sgd_clf,'saved_var/sgd_clf_binary')
else:
    sgd_clf = joblib.load('saved_var/sgd_clf_binary')    
# Predict a sample
# sample_id = 21
# plot_digit(X_train[sample_id],label=y_train[sample_id])
# print(sgd_clf.predict([X_train[sample_id]]))

#Sample_id = 0 la ankle boot
#Va no da du doan la False => chinh xac
#id = 21 la trouser
#Va no du doan Falswe => chinh xac

# PERFORMANCE MEASURE
# Accuracies (cross_val_score) sgd

from sklearn.model_selection import cross_val_score

#Chung ta do do chinh xac cua class SGDClassifier
if new_run:
    accuracies = cross_val_score(sgd_clf, X_train, y_train_Tshirt, cv=3, scoring="accuracy")
    joblib.dump(accuracies,'saved_var/sgd_clf_binary_acc')
else:
    accuracies = joblib.load('saved_var/sgd_clf_binary_acc')

# print('_________ ACCURACIES of SGD CLASSIFIER__________')
# print(accuracies)

#Co the thay, xac suat chinh xac cua SGD la cao (>90%)

# Accuracies with Dump Classifier
#Chung ta se tao ra class DumpClassifier
#Sau do tinh accuracies cua DumpClassifier, va so sanh voi SGDClassifier
from sklearn.base import BaseEstimator
class DumpClassifier(BaseEstimator): # always return False (not-5 label)
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
model_dump = DumpClassifier()
accuracies = cross_val_score(model_dump,X_train,y_train_Tshirt,cv=3,scoring='accuracy')

# print('_________ ACCURACIES of DUMP CLASSIFIER__________')
# print(accuracies)

#Co the thay, DumpClassifier cung cho xac suat chinh xac kha cao
#Nhung do chinh xac lai khong bang SGDClassifier


#---------------------------------------------------#
from sklearn.model_selection import cross_val_predict 

if new_run:
    y_train_pred_sgd = cross_val_predict(sgd_clf, X_train, y_train_Tshirt, cv=3)
    joblib.dump(y_train_pred_sgd,'saved_var/y_train_pred_sgd')
    y_train_pred_dump = cross_val_predict(model_dump, X_train, y_train_Tshirt, cv=3)
    joblib.dump(y_train_pred_dump,'saved_var/y_train_pred_dump')  
else:
    y_train_pred_sgd = joblib.load('saved_var/y_train_pred_sgd')
    y_train_pred_dump = joblib.load('saved_var/y_train_pred_dump')

#---------------------------------------------------#

#Chung ta se tinh confusion_matrix cho SGDClassifier va DumpClassifier

from sklearn.metrics import confusion_matrix  
confusion_matrix_sgd = confusion_matrix(y_train_Tshirt, y_train_pred_sgd) # row: actual class, column: predicted class. 
print('_________ CONFUSION MATRIX of SGD CLASSIFIER__________')
print(confusion_matrix_sgd)
# confusion_matrix_dump = confusion_matrix(y_train_Tshirt, y_train_pred_dump) # row: actual class, column: predicted class. 
# print('_________ CONFUSION MATRIX of DUMP CLASSIFIER__________')
# print(confusion_matrix_dump)
# Perfect prediction: zeros off the main diagonal 
# y_train_perfect_predictions = y_train_Tshirt  # pretend we reached perfection
# confusion_matrix_perfect=confusion_matrix(y_train_Tshirt, y_train_perfect_predictions)
# print('_________THE BEST CONFUSION MATRIX__________')
# print(confusion_matrix_perfect)

#Co the thay, SGDClassifier cho ra do chinh xac cao, khi ma duong cheo chinh cua matrix
#kha lon
#Con DumpClassifier khong chinh xac bang 

#Chung ta se tinh precision va recall cua SGDClassifier
# # 3.4. Precision and recall (>> see slide)
from sklearn.metrics import precision_score, recall_score
# print("______________ PRECISION & RECALL OF SGD CLASSIFIER_____________")
# print(precision_score(y_train_Tshirt, y_train_pred_sgd))
# print(recall_score(y_train_Tshirt, y_train_pred_sgd))

#------------------------------------#
#Chung ta tinh f1_score cho SGDCLassifier
from sklearn.metrics import f1_score
f1_score(y_train_Tshirt, y_train_pred_sgd)

# 3.6. Precision/Recall tradeoff (>> see slide) 
# 3.6.1. Try classifying using some threshold (on score computed by the model)  
sample_id = 1
y_score = sgd_clf.decision_function([X_train[sample_id]]) # score by the model
#Score cua sample_id = 11 la -21163, kha nang cao se la False
#Chung ta ve thu id=11 la cai gi => ankle boot => du doan chinh xac
threshold = 0
y_some_digit_pred = (y_score > threshold)
# Raising the threshold decreases recall
threshold = 10000
y_some_digit_pred = (y_score > threshold)  

if new_run:
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_Tshirt, cv=3, method="decision_function")
    joblib.dump(y_scores,'saved_var/y_scores')
else:
    y_scores = joblib.load('saved_var/y_scores')

# Plot precision,  recall curves
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_Tshirt, y_scores)
let_plot = True
# if let_plot:
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
#     plt.legend() 
#     plt.grid(True)
#     plt.xlabel("Threshold")   

# # Plot a threshold
# thres_value = 1000
# thres_id = np.min(np.where(thresholds >= thres_value))
# precision_at_thres_id = precisions[thres_id] 
# recall_at_thres_id = recalls[thres_id] 
# if let_plot:
#     plt.plot([thres_value, thres_value], [0, precision_at_thres_id], "r:")    
#     plt.plot([thres_value], [precision_at_thres_id], "ro")                            
#     plt.plot([thres_value], [recall_at_thres_id], "ro")            
#     plt.text(thres_value+500, 0, thres_value)    
#     plt.text(thres_value+500, precision_at_thres_id, np.round(precision_at_thres_id,3))                            
#     plt.text(thres_value+500, recall_at_thres_id, np.round(recall_at_thres_id,3))     
#     plt.show()

# 3.6.3. Precision vs recall curve (Precision-recall curve)
# if let_plot:         
#     plt.plot(recalls, precisions, "b-", linewidth=2)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.axis([0, 1, 0, 1])
#     plt.grid(True)
#     plt.title("Precision-recall curve (PR curve)")
#     plt.show()

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_Tshirt, y_scores)

# 3.7.2. Compute FPR, TPR for a random classifier (make prediction randomly)
from sklearn.dummy import DummyClassifier
# dmy_clf = DummyClassifier(strategy="uniform")
# y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_Tshirt, cv=3, method="predict_proba")
# y_scores_dmy = y_probas_dmy[:, 1]
# fprr, tprr, thresholdsr = roc_curve(y_train_Tshirt, y_scores_dmy)

# # 3.7.3. Plot ROC curves
# if let_plot:
#     plt.plot(fpr, tpr, linewidth=2)
#     plt.plot(fprr, tprr, 'k--') # random classifier
#     #plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal: random classifier
#     plt.legend(['SGDClassifier','Random classifier'])
#     plt.grid(True)        
#     plt.axis([0, 1, 0, 1])                                    
#     plt.xlabel('False Positive Rate')  
#     plt.ylabel('True Positive Rate (Recall)')    
#     plt.show()


# from sklearn.metrics import roc_auc_score
# roc_auc = roc_auc_score(y_test_Tshirt, y_scores)

#Chung ta se dung RandomForestClassifier de so sanh voi SGDClassifer
from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(n_estimators=100)

if new_run:
    random_clf.fit(X_train,y_train_Tshirt)
    joblib.dump(random_clf,'saved_var/random_clf_binary')
else:
    random_clf = joblib.load('saved_var/random_clf_binary')
    
if new_run:
    accuracies = cross_val_score(random_clf,X_train,y_train_Tshirt,cv = 3, scoring = 'accuracy')
    joblib.dump(accuracies,'saved_var/random_clf_binary_acc')
else:
    accuracies = joblib.load('saved_var/random_clf_binary_acc')
  
# print('______________________ACCURACIES OF RANDOM FOREST CLASSIFIER________________')  
# print(accuracies)

#Chung ta co the thay rang, accuracies cua RandomForest tot hon SGD

if new_run:
    y_scores_random = cross_val_predict(random_clf,X_train,y_train_Tshirt,cv=3,method = 'predict_proba')
    joblib.dump(y_scores_random,'saved_var/random_clf_binary_yscores')
else:
    y_scores_random = joblib.load('saved_var/random_clf_binary_yscores')
    
# print(y_scores_random)
if new_run:
    y_train_pred_random = cross_val_predict(random_clf,X_train,y_train_Tshirt,cv=3)
    joblib.dump(y_train_pred_random,'saved_var/y_train_pred_random')
else:
    y_train_pred_random = joblib.load('saved_var/y_train_pred_random')
    
#Chung ta se so sanh precision va recall giua SGD va Random

# print("______________ PRECISION & RECALL OF RANDOM FOREST CLASSIFIER_____________")
# print(precision_score(y_train_Tshirt, y_train_pred_random))
# print(recall_score(y_train_Tshirt, y_train_pred_random))

#Precision va recall Random tot hon rat nhieu so voi SGD


#Chung ta se so sanh confusion_matrix giua Random va SGD
confusion_matrix_random = confusion_matrix(y_train_Tshirt, y_train_pred_random) # row: actual class, column: predicted class. 
print('_________ CONFUSION MATRIX of RANDOM FOREST CLASSIFIER__________')
print(confusion_matrix_random)

#Nhin vao ket qua, RandomForestClassifier tot hon so voi SGDClassifier