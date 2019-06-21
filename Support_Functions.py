import numpy as np
import matplotlib.image as mpimg
import scipy.io
import scipy.misc
import random
from PIL import Image

def normalize_row(x):
    ss = np.multiply(x, x)
    ss = np.sum(ss, axis=1)
    ss = np.sqrt(ss)
    data = (x.T * (1 / ss)).T
    return data


def predict (logits):
    zero_count = np.count_nonzero(logits, axis=1)
    invalid_list = np.nonzero(zero_count == 0)
    predictions = np.argmax(logits, axis=1)
    predictions[invalid_list] = -1
    return predictions


def compute_acc (predictions, labels):
    # predictions = predict(logits)
    correct = float(np.count_nonzero(predictions == labels))
    return correct / len(predictions)


def compute_mAP(predictions, labels):
    # predictions = predict(logits)
    nc = np.max(labels)+1
    nc0 = np.min(labels)
    avg_sum = 0
    for i in range (nc0,nc):
        c_list = np.nonzero(labels == i)
        predictions_c = predictions[c_list]
        labels_c = labels[c_list]
        if len(predictions_c) ==0:
            continue
        avg_sum = avg_sum + float(np.count_nonzero(predictions_c == labels_c))/len(labels_c)
    return avg_sum/(nc-nc0)



def convert_label(file_name, num_class = 20):
    classes = 0
    class_map = {}
    labels = []
    with open(file_name) as f:
        for line in f:
            labels.append(int (line))
    return np.array(labels, dtype = np.int32)


def shuffle_data(IMG_files_train, labels_train):
    comb = list(zip(IMG_files_train, labels_train))
    random.shuffle(comb)
    IMG_files_train[:], labels_train[:] = zip(*comb)
    return IMG_files_train, labels_train

#----------------------------------------------------------------------------------------------------------------------

def Array_de_zeros(A, epsilon=1e-5):
    A[np.nonzero((A >= 0) * (A < epsilon))] = epsilon
    A[np.nonzero((A < 0) * (A > -epsilon))] = -epsilon
    return A



def load_img_as_array(file_name_list, size, channels, offset=127.5):
    ImgArray = np.zeros([len(file_name_list), size, size, channels], dtype=np.float32)
    i = 0
    for file in file_name_list:
        image = Image.open(file)
        if image.mode == 'CMYK':
            image = image.convert('RGB')

        Img = np.asarray(image)
        if len(Img.shape) < 3:
            Img_RGB = np.expand_dims(Img, axis=2)
            Img_RGB = np.append(Img_RGB, np.expand_dims(Img, axis=2), axis=2)
            Img_RGB = np.append(Img_RGB, np.expand_dims(Img, axis=2), axis=2)
            Img = Img_RGB

        if random.uniform(0, 1) > 0.5:
            Img = np.flip(Img, 1)

        ImgArray[i, :, :, :] = Img
        i = i + 1
    ImgArray = (ImgArray - offset)/offset
    return ImgArray

def load_img_as_array_augment(file_name_list, size, channels, offset=127.5):
    ImgArray = np.zeros([len(file_name_list), size, size, channels],dtype = np.float32)
    i = 0
    for file in file_name_list:
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        Img = np.asarray(image)
        if len(Img.shape) < 3:
            Img_RGB = np.expand_dims(Img, axis = 2)
            Img_RGB = np.append(Img_RGB,  np.expand_dims(Img, axis = 2), axis = 2)
            Img_RGB = np.append(Img_RGB, np.expand_dims(Img, axis=2), axis=2)
            Img = Img_RGB

        if random.uniform(0, 1) > 0.5:
            Img = np.flip(Img, 1)
        Img_resized = Img
        short_side = random.randint(256, 480)
        if Img.shape[0] < Img.shape[1]:

            # if Img.shape[0] > 256:
            #     short_side = random.randint(256, min(Img.shape[0], 480))
            # else:
            #     short_side = 256 # random.randint(Img.shape[0], 256)
            Img_resized =  scipy.misc.imresize(Img, (short_side, int(Img.shape[1]*short_side/Img.shape[0])))
        else:
            # if Img.shape[1] > 256:
            #     short_side = random.randint(256, min(Img.shape[1], 480))
            # else:
            #     short_side = 256 # random.randint(Img.shape[1], 256)
            Img_resized = scipy.misc.imresize(Img, ( int(Img.shape[0]*short_side/Img.shape[1]), short_side))


        a = random.randint(0, Img_resized.shape[0] - size-1)
        b = random.randint(0, Img_resized.shape[1]- size-1)
        Img = Img_resized[a:a+size,b:b+size,:]
        ImgArray[i, :, :, :] = Img
        i = i + 1
    ImgArray = (ImgArray - offset)/offset
    return ImgArray