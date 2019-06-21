from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import scipy.io
import os.path
import scipy.misc
import random
import threading

from CosNet import *
from cluster_DLL import *
from Support_Functions import *

class augment_batch (threading.Thread):
    def __init__(self, IMG_files, aug = True):
        threading.Thread.__init__(self)
        self.IMG_files = IMG_files
        self.aug = aug
        self.ImgArray = []
    def run(self):
        # print ('start loading batch')
        if self.aug is True:
            self.ImgArray = load_img_as_array_augment(self.IMG_files, 224, 3).astype(np.float32)
        else:
            self.ImgArray = load_img_as_array(self.IMG_files, 224, 3).astype(np.float32)
        # print('finished loading batch')

class run_train_op(threading.Thread):
    def __init__(self, train_op, sess, ImgArray, batch_labels, lr):
        threading.Thread.__init__(self)
        self.train_op = train_op
        self.sess = sess
        self.ImgArray = ImgArray
        self.batch_labels = batch_labels
        self.lr = lr
    def run(self):
        # print('start training')
        self.sess.run(self.train_op, feed_dict={Img: self.ImgArray, ll: self.batch_labels, training: True, learning_rate: self.lr})
        # print('finished training')

class run_eval_op(threading.Thread):
    def __init__(self, sess, ImgArray, batch_labels, loss, logits):
        threading.Thread.__init__(self)
        self.sess = sess
        self.ImgArray = ImgArray
        self.batch_labels = batch_labels
        self.loss_epoch = -1
        self.predictions_batch = []
        self.loss = loss
        self.logits = logits
    def run(self):
        # print('start evaluation')
        self.loss_epoch = self.sess.run(self.loss, feed_dict={Img: self.ImgArray, ll: self.batch_labels, training:False})
        logit_batch = self.sess.run(self.logits, feed_dict={Img: self.ImgArray, ll: self.batch_labels, training:False})
        self.predictions_batch = predict(logit_batch)
        # print('finished evaluation')

def train_one_epoch_parallel(training_data, train_eval = [], train = True, mini_batch = 64, lr = 0.001,
                     sess = [], ll = [], Img = [], logits = [], aug = True):
    # parallel data augmentation/loading  and net training
    # for train is True: train_op is the training operation
    # for train is False: train_op[0] is loss

    # shufling data
    IMG_files_train = training_data[0]
    labels_train = training_data[1]


    loss_epoch = 0
    train_cycle = 0

    num_imgs = len(IMG_files_train) - len(IMG_files_train) % mini_batch
    predictions = -1 * np.ones(num_imgs)
    labels_train = labels_train[0:num_imgs]
    IMG_files_train = IMG_files_train[0:num_imgs]

    #load first mini_batch
    if aug == True:
        ImgArray = load_img_as_array_augment(IMG_files_train[0:mini_batch], 224, 3)
    else:
        ImgArray = load_img_as_array(IMG_files_train[0:mini_batch],224,3)
    batch_labels = labels_train[0:mini_batch]
    # other mini batches
    for i in range(mini_batch, len(IMG_files_train) - 1, mini_batch):

        aug_thread = augment_batch(IMG_files_train[i: i + mini_batch], aug = aug)

        if train is True:
            train_thread = run_train_op( train_op, sess, ImgArray, batch_labels, lr)
        else:
            train_thread = run_eval_op(sess, ImgArray, batch_labels, train_eval[0], logits)

        aug_thread.start()
        train_thread.start()

        aug_thread.join()
        train_thread.join()

        ImgArray = aug_thread.ImgArray
        if train is False:
            predictions[i-mini_batch:i] = train_thread.predictions_batch
            loss_epoch += train_thread.loss_epoch
        batch_labels = labels_train[i: i + mini_batch]
        train_cycle += 1

    # last mini_batch
    if train is True:
        sess.run(train_op, feed_dict={Img: ImgArray, ll: batch_labels, training:True, learning_rate: lr})

    else:
        loss_epoch += sess.run(train_eval[0], feed_dict={Img: ImgArray, ll: batch_labels, training:False})
        logit_batch = logits.eval(feed_dict={Img: ImgArray, ll: batch_labels, training:False})
        predictions[num_imgs-mini_batch:num_imgs] = predict(logit_batch)
        return loss_epoch / train_cycle, compute_acc(predictions, labels_train), compute_mAP(predictions, labels_train)

def incrementCosCovLayer(IMG_files, PreLayer, cc, pSize = 5, mini_batch = 64, Img = [], max_knum=1024):
    # cc: cluster handle with all set parameters
    # IMG_files [list]: full path to each image
    # return: [tf tensor] a covmap of next layer
    nc = 0
    p = tf.image.extract_image_patches(PreLayer, [1, pSize, pSize, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        post_max_training_epoch = 1000 # when the number of cluster increased to max_knum, keep training for this number of epochs
        for i in range(0, len(IMG_files) - 1, mini_batch):
            last_indx = i + mini_batch
            if (last_indx >= len(IMG_files)):
                break

            for j in range (2):
                ImgArray =  load_img_as_array_augment(IMG_files[i: last_indx], 224, 3)
                patches = sess.run(p, feed_dict={Img: ImgArray})
                patches = np.reshape(patches,
                                     [patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3]])
                idx = np.random.permutation(patches.shape[0])[0:(last_indx - i + 1) * 500]
                patches = patches[idx,:]
                patches = patches.astype(np.double)
                patches = normalize_row(patches)
                cc.feed(patches)
                nc = lib.num_cluster(cc.handle)

                print("minibatch ", i, ";  nc ", nc)
            if nc >= max_knum:
                post_max_training_epoch-=1
                print("last " + str(post_max_training_epoch)+" epochs")
            if post_max_training_epoch == 0:
                break
            if i % 128*100 == 0 and i > 0:
                c = cc.get().astype(np.float32)
                np.save("temp.npy", c)
                print("kernels saved to temp.npy")

    c = cc.get().astype(np.float32)
    layer = netlayer(type='cos', pSize = pSize, kernels=c)
    return layer

def load_kernels(file_name, gamma = 0.05, max_knum = 1024):
    kernel_array = np.load(file_name)
    if kernel_array.shape[0]>max_knum:
        kernel_array = kernel_array[0:max_knum,:,:,:]
    kernel_list = []
    for i in range (kernel_array.shape[0]):
        kernel_list.append(gamma*kernel_array[i,:,:,:].astype(np.float32))
    return kernel_list



def coslayer(input, pSize1, kernel_file_name, pf, th, gamma=0.05, pretrain=False, Img=[], max_knum=1024,
             stride = 1, IMG_files=[], mini_batch = 64):

    channel = int(input.get_shape()[3])
    if pretrain:
        cc1 = clust(pf, th, channel*pSize1*pSize1, max_knum)
        layer1= incrementCosCovLayer(IMG_files, input, cc1, pSize = pSize1, mini_batch = mini_batch, Img = Img, max_knum=max_knum)
        np.save(kernel_file_name, np.array(layer1.kernels))

    layer1_kernels = load_kernels(kernel_file_name, gamma = gamma, max_knum = max_knum)
    conv_map1, convkernel1 = CosCovLayer(input, layer1_kernels, stride = stride, padding = 'SAME')
    return conv_map1


#############################################################################################################
#############################################################################################################
#############################################################################################################
augment = True
if augment:
    img_path = 'F:\\ILSVRC_train\\'
else:
    img_path = 'C:\\Users\\wangz\\MyDatabase\\LSVRC\\'
num_class = 10
IMG_files_train = []
with open('C:\\Users\\wangz\\iCloudDrive\\Matlab\\ImageNet\\imgs_LSVRC_'+str(num_class)+'_train.txt') as f:
    for line in f:
        IMG_files_train.append(img_path+line[:-1])
labels_train = convert_label('C:\\Users\\wangz\\iCloudDrive\\Matlab\\ImageNet\\labels_train_'+str(num_class)+'.txt', num_class= num_class)
IMG_files_val = []
with open('C:\\Users\\wangz\\iCloudDrive\\Matlab\\ImageNet\\imgs_LSVRC_'+str(num_class)+'_val.txt') as f:
    for line in f:
        IMG_files_val.append(img_path+line[:-1])
labels_val = convert_label('C:\\Users\\wangz\\iCloudDrive\\Matlab\\ImageNet\\labels_val_'+str(num_class)+'.txt',num_class= num_class)

IMG_files_train, labels_train = shuffle_data(IMG_files_train, labels_train)
IMG_files_val, labels_val = shuffle_data(IMG_files_val, labels_val)
#############################################################################################################
gamma = [0.05, 0.05, 0.05, 0.05]
pool_size = [2,2,2,2]
init_value = []

mini_batch = 128
lr = 1e-3

tf.reset_default_graph()
resinitializer = tf.initializers.random_uniform(minval = -1e-5, maxval = 1e-5)
# resinitializer=None
training = tf.placeholder(tf.bool, name="is_train")


Img = tf.placeholder(tf.float32, shape = (mini_batch, 224, 224, 3), name = "mini_batch_imgs")

#-----------------------------------------------Layer 1-----------------------------------------------------------------
# Conv 1_1 layer

pSize1 = 7
kernel_file_name = 'layer1_kernels_'+str(pSize1)+'.npy'
conv_map1 = coslayer(Img, pSize1, kernel_file_name, 0.01, 0.85, gamma=0.05, pretrain = False, Img=Img, max_knum=64,
         stride=2, IMG_files = IMG_files_val+IMG_files_train, mini_batch = mini_batch)
conv1 = tf.nn.leaky_relu(conv_map1)
maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[pool_size[0], pool_size[0]], strides=2, padding='SAME')
res1 = tf.nn.leaky_relu(resblock(maxpool1, 3, 64,  resinitializer, training, stride = (1,1), name = 'res_block_1_1'))

kernel_file_name = 'layer2_kernels_3.npy'
conv_map2= coslayer(res1, 3, kernel_file_name, 0.005, 0.925, gamma=0.05, pretrain = False, Img=Img, max_knum=128,
         stride=1, IMG_files = IMG_files_val+IMG_files_train, mini_batch = mini_batch)
conv2 = tf.nn.leaky_relu(conv_map2)
maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[pool_size[1], pool_size[1]], strides=2, padding='SAME')
res2 = tf.nn.leaky_relu(resblock(maxpool2, 3, 128,  resinitializer, training, stride = (1,1), name = 'res_block_2_1')) 

kernel_file_name = 'layer3_kernels_3.npy'
conv_map3= coslayer(res2, 3, kernel_file_name, 0.001, 0.95, gamma=0.05, pretrain = False, Img=Img, max_knum=256,
         stride=1, IMG_files = IMG_files_val+IMG_files_train, mini_batch = mini_batch)
conv3 = tf.nn.leaky_relu(conv_map3)
maxpool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[pool_size[2], pool_size[2]], strides=2, padding='SAME') 

res3 = tf.nn.leaky_relu(resblock(maxpool3, 3, 256,  resinitializer, training, stride = (1,1), name = 'res_block_4_1') )

kernel_file_name = 'layer4_kernels_3.npy'
conv_map4= coslayer(res3, 3, kernel_file_name, 0.00025, 0.99, gamma=0.05, pretrain = False, Img=Img, max_knum=512,
         stride=1, IMG_files =  IMG_files_val+ IMG_files_train, mini_batch = mini_batch)
conv4 = tf.nn.leaky_relu(conv_map4)
maxpool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[pool_size[3], pool_size[3]], strides=2, padding='SAME')
res4 = tf.nn.leaky_relu(resblock(maxpool4, 3, 512,  resinitializer, training, stride = (1,1), name = 'res_block_6_1'))

avgpool = tf.layers.average_pooling2d(inputs=res4, pool_size=[7, 7], strides=1)
pool_Flat = tf.contrib.layers.batch_norm(tf.layers.flatten(avgpool))
logits = tf.layers.dense(pool_Flat, num_class, name = 'logits', activation=tf.nn.leaky_relu)

ll = tf.placeholder (tf.int32, name = "mini_batch_labels", shape = mini_batch)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits (labels=ll, logits = logits)
loss = tf.reduce_mean(xentropy, name = "loss")

learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(epsilon=1e-4, learning_rate = learning_rate)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group([train_op, update_ops])

with tf.name_scope ("eval"):
    correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), ll, 1)
    correct5 = tf.nn.in_top_k(tf.cast(logits, tf.float32), ll, 5)
    accuracy = tf.reduce_mean(tf.cast (correct, tf.float32))
    accuracy5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

result_file = open('recording_'+str(num_class)+'.txt','w+')
result_file.close()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(150):

        IMG_files_train, labels_train = shuffle_data(IMG_files_train, labels_train)
        IMG_files_val, labels_val = shuffle_data(IMG_files_val, labels_val)

        train_one_epoch_parallel([IMG_files_train, labels_train], train_eval=[train_op], train=True,
                                 mini_batch=mini_batch, lr=lr, sess=sess, ll=ll, Img=Img, logits=logits, aug=augment)

        loss_epoch_train, acc_train, mAP_train = train_one_epoch_parallel(
            [IMG_files_train[0:4096], labels_train[0:4096]],
            train_eval=[loss, accuracy, accuracy5],
            train=False, mini_batch=mini_batch, lr=lr,
            sess=sess, ll=ll, Img=Img, logits=logits, aug=augment)

        loss_epoch_val, acc_val, mAP_val = train_one_epoch_parallel(
            [IMG_files_val, labels_val],
            train_eval=[loss, accuracy,
                     accuracy5],
            train=False,
            mini_batch=mini_batch, lr=lr,
            sess=sess, ll=ll, Img=Img,
            logits=logits, aug=False)

        print(epoch, " ",
              (acc_train * 100), " ", loss_epoch_train, " ",
              (acc_val * 100), " ", loss_epoch_val)

        result_file = open('recording_' + str(num_class) + '.txt', 'a+')
        epoch_result = str(epoch) + " " + \
                       str(acc_train * 100) + " " + str(loss_epoch_train) + " " + \
                       str(acc_val * 100) + " " + str(loss_epoch_val) + " " '\n'
        result_file.write(epoch_result)
        result_file.close()







