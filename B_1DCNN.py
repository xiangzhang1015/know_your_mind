import tensorflow as tf
import scipy.io as sc
import numpy as np
import pickle
import random
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from sklearn.metrics import classification_report
from scipy import stats

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#
len_sample=1
full=7000
len_a=full/len_sample  # 6144 class1
label0=np.zeros(len_a)
label1=np.ones(len_a)
label2=np.ones(len_a)*2
label3=np.ones(len_a)*3
label4=np.ones(len_a)*4
label5=np.ones(len_a)*5
label6=np.ones(len_a)*6
label7=np.ones(len_a)*7
label=np.hstack((label0,label1,label2,label3,label4,label5,label6,label7))
label=np.transpose(label)
label.shape=(len(label),1)
print label
time1 =time.clock()
# feature = sc.loadmat("eeg_close_ubicomp_8sub.mat")  # EID_M, with three trials, 21000 samples per sub
# # all = feature['eeg_close_ubicomp_8sub']  /home/xiangzhang/matlabwork/
# feature = sc.loadmat("eeg_close_8sub_1file.mat")  # EID-S, with 1 trial, 7000 samples per subject
# all = feature['eeg_close_8sub_1file']
#
# n_fea=14
# all = all[0:full*8, 0:n_fea]

# EEG-S dataset is a subset of EEG_ID_label6.mat. 1 trial, 7000 samples per sub
# /home/xiangzhang/matlabwork/eegmmidb/

feature = sc.loadmat("EEG_ID_label6.mat")  # 1trial, 13500 samples each subject
all = feature['EEG_ID_label6']
n_fea = 64
all = all[0:21000*8, 0:n_fea]
print all.shape

a1 = all[0:7000]  # select 7000 samples from 135000
for i in range(2,9):
    b = all[13500*(i-1):13500*i]
    c = b[0:7000]
    print c.shape
    a1 = np.vstack((a1, c))
    print i, a1.shape
all = a1
time3=time.clock()
# print 'PD time',time3-time2
all=np.hstack((all,label))
print all.shape
n_classes = 8
n_fea = all.shape[-1]-1
np.random.shuffle(all)
all = all[0:int(5000*1)]
#### all = np.hstack((all[:, 0:int(n_fea*0.8)], all[:, n_fea:n_fea+1]))



# # PAMAP2 120,000 samples for 6 subjects., 20,000 samples for each sub. 51 features, 8 activities(0-7)
# feature = sc.loadmat("/home/xiangzhang/scratch/AR_6p_8c.mat")
# all = feature['AR_6p_8c']
# # the subset from the first sub, the first 17 features in the hand IMU.
# all = np.hstack((all[0:200000, 0:17], all[0:200000, 51:52]))
# np.random.shuffle(all)
# all = all[0:10000]
# n_classes =8

# RFID, local dataset, 3100 samples, 1 subject, 21 classes. (1-21), n_classes =22
# feature = sc.loadmat("/home/xiangzhang/scratch/rssi_nonmix_all.mat")
# all = feature['rssi_nonmix_all']
# np.random.shuffle(all)
# n_classes =22

# SPAR, (smartphone for AR) 60,000 samples, 9 features, 6 labels (0-5), n_classes=6
# feature = sc.loadmat("/home/xiangzhang/scratch/Phone_arm_6class.mat")
# all = feature['Phone_arm_6class']
# np.random.shuffle(all)
# n_classes =6

# HAPT, (smartphone for AR) 7767 samples, 567 features, 12 labels (0-5), n_classes=12
# feature = sc.loadmat("/home/xiangzhang/scratch/HAPT_561feature.mat")
# all = feature['Xtrain']
# all=np.hstack((all[0:7000, 0:40], all[0:7000, 561:562]))
# np.random.shuffle(all)
# n_classes =134

# unimib SHAR, (smartphone for AR) 11771 samples, 453 features, the 454th column is activity ID
# 17labels (1-17), n_classes =18
# feature = sc.loadmat("/home/xiangzhang/scratch/unimib.mat")
# all = feature['unimib']
# all=np.hstack((all[0:10000, 0:30], all[0:10000, 453:454]))
# np.random.shuffle(all)
# n_classes =18

# WISDM_36sub_6class_1million.mat, 36sub,1,098,207 samples. more than 1million samples. 6 classes [0-5] n_classes =6
# WISDM_sub33.mat, the 33th sub subset. 28,000 samples, 3 features, the 4th is label. 6 classes [0-5] n_classes =6
# feature = sc.loadmat("/home/xiangzhang/scratch/WISDM_sub33.mat")
# all = feature['WISDM_sub33']
# all = all[:, 0:4]   # this 5th column is the No. of subjects. is a constant 33. the 4th column is the class label.
# np.random.shuffle(all)
# n_classes =6

# ARS wearable, 9 features, 3969 samples. label :1 is abnormal, 0 is normal
# feature = sc.loadmat("/home/xiangzhang/Downloads/Data/ARS DLR Data Set/ARS_abnormal_1sub.mat")
# all_normal = feature['ARS_abnormal_1sub']
# all_normal = all_normal[0:3900]  # 693 abnormal
# feature_fall = sc.loadmat("/home/xiangzhang/Downloads/Data/ARS DLR Data Set/ARS_abnormal_1sub.mat")
# all_fall = feature_fall['ARS_abnormal_1sub'] # 3402 samples
# all_ = np.vstack((all_normal, all_fall))
# n_fea = all_.shape[-1]-1
# feature_normalized=preprocessing.normalize(all_[:, 0:n_fea], axis=0)
# all_ = np.hstack((feature_normalized, all_[:, n_fea:n_fea+1]))
# np.random.shuffle(all_)
# n_classes = 2
# all = all_


# #########  EEG THU seizure
# import pyedflib
# import numpy as np
# # /home/xiangzhang/Downloads/Data/EEG_TUH/Seizure/train/01_tcp_ar/00000018/s02_2012_10_11/
# file_name='00000018_s02_a00.edf'
# f = pyedflib.EdfReader(file_name)
# n = f.signals_in_file
# signal_labels = f.getSignalLabels()
# data=f.readSignal(0)
# for i in np.arange(1, 26):
#     h = f.readSignal(i)
#     data = np.vstack((data, h))
# data = np.transpose(data) # he first 21 column is 21 EEG channels. Checked it by the edfbrower.
# # The sampling rate is 250 Hz, totally 1203 seconds, 300,750 samples.
# print data.shape,
# train = data[250*100:250*124, 0:21] # normal
# len_a =250*124-250*100
# label_train = np.zeros(len_a)
# label_test = np.ones(len_a)
# label_train.shape = [len_a, 1]
# label_test.shape = [len_a, 1]
# train = np.concatenate((train, label_train), axis=1)
# test = data[250*263:250*287, 0:21] # abnormal
# test = np.concatenate((test, label_test), axis=1)
# all_ = np.vstack((train, test))
# n_fea = all_.shape[-1]-1
# feature_normalized=preprocessing.normalize(all_[:, 0:n_fea], axis=0)
# all_ = np.hstack((feature_normalized, all_[:, n_fea:n_fea+1]))
#
# np.random.shuffle(all_)
# n_classes = 2
# all = all_





## EID, 168000 samples, 8 subjects, (1-8), n_classes =9
# feature = sc.loadmat("eeg_close_ubicomp_8sub.mat")
# all = feature['eeg_close_ubicomp_8sub']
#
# data_f1=[]
#  # EEG Delta pattern decomposition
# for i in range(all.shape[1]-1):
#     x = all[:, i]
#     fs = 128.0
#     lowcut = 0.5
#     highcut = 4.0
#     y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
#     data_f1.append(y)
# data_f1=np.array(data_f1)
# data_f1=np.transpose(data_f1)
# print 'data_f1.shape',data_f1.shape
# all= np.hstack((data_f1, all[:, 14:15]))
# n_classes =9
# np.random.shuffle(all)

## eegmmidb 8 person, ID
# feature = sc.loadmat("EEG_ID_8person.mat")
# all = feature['EEG_ID_8person']
#
# data_f1=[]
#  # EEG Delta pattern decomposition
# for i in range(all.shape[1]-1):
#     x = all[:, i]
#     fs = 128.0
#     lowcut = 0.5
#     highcut = 3.5
#     y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
#     data_f1.append(y)
# data_f1=np.array(data_f1)
# data_f1=np.transpose(data_f1)
# print 'data_f1.shape',data_f1.shape
# all= np.hstack((data_f1, all[:, 64:65]))
# n_classes =9
# np.random.shuffle(all)



## KDD_data.mat, 20sub, each sub 28,000 samples, 64 features. 5 labels (1-5). n_classes =6
# feature = sc.loadmat("/home/xiangzhang/scratch/KDD_data.mat")
# all = feature['KDD_data']
# all = all[28000*2:28000*3]
# n_fea = all.shape[-1]-1
# # np.random.shuffle(all)
# # all = all[0:int(28000*0.8)]
# n_classes =6


## emotiv class recognition data, 34560 per sub, totally 7 sub. 14 features,
# the 15th is action ID, 1-6, 6 IDs.
## the 16th column is sub ID. 1-7, 7 subjects
# feature = sc.loadmat("/home/xiangzhang/scratch/emotiv_7sub_5class.mat")
# all = feature['emotiv_7sub_5class']
# all = all[34560*0:34560*1, 0:15] ## ID identification
# n_classes =7
# np.random.shuffle(all)


# THU pickle file /home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/IJCAI_KDD/
# import pickle
# all_ = pickle.load(open("EEG_THU_seizure.p", "rb" ) )
# np.random.shuffle(all_)
# n_classes = 2
# n_fea = all_.shape[-1]-1
# all = all_


##segmentation
print 'all', all.shape
len_sample = 1
data_size=all.shape[0]
no_fea= all.shape[1] - 1
F_ = all[:, 0:no_fea]
L_ = all[:, no_fea:no_fea+1]
#z-score scaling
F_=preprocessing.scale(F_)

print F_.shape, data_size, len_sample, no_fea
F = F_.reshape([data_size/len_sample, len_sample*no_fea])
L = L_.reshape([data_size/len_sample, len_sample])
# calculate the mode of L
L, _ =stats.mode(L, axis=1)
L = np.array(L)
print L.shape

all = np.hstack((F, L))

# data batach seperation
np.random.shuffle(all)  # mix eeg_all
# update the no_fea, ex: 14 -> 140
no_fea= all.shape[1] - 1
data_size=all.shape[0]  # update data_size
feature_all = all[:, 0:no_fea]
print no_fea, all[:, -1]



# use the first subject as testing subject
train_data=all[0:data_size*0.9] #1 million samples
test_data=all[data_size*0.9:data_size]


feature_training =train_data[:,0:no_fea]
# feature_training =feature_training.reshape([data_size*0.9,n_steps,no_fea/n_steps])
print feature_training.shape, data_size, len_sample, no_fea/len_sample
# feature_training =feature_training.reshape([data_size*0.9,len_sample, no_fea/len_sample])


feature_testing =test_data[:,0:no_fea]
# feature_testing =feature_testing.reshape([data_size*0.1,n_steps,no_fea/n_steps])
# feature_testing =feature_testing.reshape([data_size*0.1, len_sample, no_fea/len_sample])
label_training =train_data[:,no_fea:no_fea+1]
print 'label shape', label_training.shape
label_training =one_hot(label_training)
label_testing =test_data[:,no_fea:no_fea+1]
print 'label shape', label_testing.shape
label_testing =one_hot(label_testing)
print 'label shape', label_testing.shape


print all.shape
#batch split

a=feature_training
b=feature_testing
# nodes= 164
lameda=0.001
lr=0.001
fg = 0.3

batch_size=int(data_size*0.1)
train_fea=[]
n_group=9
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)
keep=1

# the CNN code
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def max_pool_2x2(x):
#     # stride [1, x_movement, y_movement, 1]
#     return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
def max_pool_1x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1, 2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, no_fea]) # 1*64
ys = tf.placeholder(tf.float32, [None, n_classes])  # 2 is the classes of the data
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, no_fea, 1])
print(x_image.shape)  # [n_samples, 28,28,1]

depth_1 = 10
## conv1 layer ##
W_conv1 = weight_variable([1, 2, 1, depth_1]) # patch 5x5, in size is 1, out size is 8
b_conv1 = bias_variable([depth_1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 1*64*2
# h_pool1 = max_pool_1x2(h_conv1)                          # output size 1*32x2

## conv2 layer ##
# depth_2 = 80
# W_conv2 = weight_variable([2,2, depth_1, depth_2]) # patch 5x5, in size 32, out size 64
# b_conv2 = bias_variable([depth_2])
# h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) # output size 1*32*64
# h_pool2 = max_pool_1x2(h_conv2)

size2 = 164
## fc1 layer ##
input_shape = (no_fea)*depth_1
W_fc1 = weight_variable([input_shape, size2])
b_fc1 = bias_variable([size2])
h_pool2_flat = tf.reshape(h_conv1, [-1, input_shape])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([size2, n_classes])
b_fc2 = bias_variable([n_classes])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print "prediction shape", prediction

# the error between prediction and real data
l2 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))+l2   # Softmax loss
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) # learning rate is 0.0001

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

start=time.clock()
step = 1
while step < 20:
    for i in range(n_group):
        sess.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob:keep})
    if step % 10 == 0:

        cost=sess.run(cross_entropy, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
        t1 = time.clock()
        acc = compute_accuracy(train_fea[i], train_label[i])
        t2 = time.clock()
        print 'testing time' , t2-t1
        print('step:',step,',train acc', acc,
              ',test acc',compute_accuracy(b, label_testing),', the cost is', cost)

    step+=1
B = sess.run(h_fc1, feed_dict={xs: train_fea[0], ys: train_label[0], keep_prob: keep})
for i in range(1, n_group):
    D = sess.run(h_fc1, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob: keep})
    B = np.vstack((B, D))
B = np.array(B)
print B.shape
Data_train = B  # Extracted deep features
Data_test = sess.run(h_fc1, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
l_p = sess.run(prediction, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
print(classification_report(np.argmax(label_testing, 1), np.argmax(l_p, 1), digits=4))

# # ### KNN
# time1 = time.clock()
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(Data_train, np.argmax(label_training,1))
time2 = time.clock()
label_predict = neigh.predict(Data_test)
time3 = time.clock()
knn_result = neigh.score(Data_test, np.argmax(label_testing,1))
print knn_result
# time3 = time.clock()
# print "training time", time2 - time1
print "testing time", time3 - time2

print(classification_report(label_predict, np.argmax(label_testing,1), digits=4))

# from sklearn.metrics import confusion_matrix
# import pickle
# cm = confusion_matrix(np.argmax(label_testing,1), label_predict)
# pickle.dump(cm, open('/home/xiangzhang/scratch/results/Intention_emotiv_cm.p',"wb"))
#
# y_yes = np.argmax(label_testing, 1)
# pickle.dump(y_yes, open('/home/xiangzhang/scratch/results/Intention_emotiv_y_yes.p',"wb"))
#
# y_score = l_p
# pickle.dump(y_score, open('/home/xiangzhang/scratch/results/Intention_emotiv_y_score.p',"wb"))
# print "cm, y_yes, y_sore saved"

# visualization
# h_conv1 = sess.run(h_conv1, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
# h_fc1 = sess.run(h_fc1, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
# pickle.dump(test_data, open('/home/xiangzhang/scratch/results/visualization_TUH_original.p',"wb"))
# pickle.dump(h_conv1, open('/home/xiangzhang/scratch/results/visualization_TUH_mapping.p',"wb"))
#
# pickle.dump(h_fc1, open('/home/xiangzhang/scratch/results/visualization_TUH_spatialfeature.p',"wb"))


