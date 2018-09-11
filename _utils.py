# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
import os
import numpy as np
import pandas as pd
import random
import cv2
from scipy.ndimage.interpolation import shift, rotate
from skimage.morphology import label
import re
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm




#####################
## Encode - Decode ##
#####################
# https://www.kaggle.com/kmader/baseline-u-net-model-part-1
def rle_encode(arr_mask):
    """
    arr_mask: 2-dim numpy array for a single mask, 1 - mask, 0 - background
    Returns a single encoded string for the mask
    """
    pixels = arr_mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img):
    """ find all connected regions and encode them separately """
    # http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.label
    labels = label(img) # background=0
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]


def rle_decode(rle_mask, shape=(768, 768)):
    """
    rle_mask: an encoded string for a single mask
    shape: (height,width) of array to return 
    Returns 2-dim numpy array, 1 - mask, 0 - background
    """
    if not rle_mask or len(rle_mask) == 0:
        return np.zeros(shape, dtype=np.int16)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def rle_decode_all(rle_mask_list):
    """
        there might be multile ships or multiple rle_mask for each image.
        there is a rle_mask_list for each image.
        decode all rle_mask into a SINGLE 2-dim numpy array with shape=(768, 768)

        rle_mask_list: list of strings for all masks in the same image. If no mask, rle_mask_list=[].
    """

    all_masks = np.zeros((768, 768), dtype = np.int16)
    for mask in rle_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    # return np.expand_dims(all_masks, -1)
    return all_masks






##############################
##### Image Augmentation #####
##############################
# https://github.com/vxy10/ImageAugmentation
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
def augment_brightness_camera_images(img):
    img = cv2.cvtColor(np.float32(img),cv2.COLOR_RGB2HSV)
    random_bright = 1.0 + 0.2*np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def transform_image(img_x, img_y,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols = img_y.shape[:2]
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1) # rotation about the center by angle ang_rot without scaling

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]]) # translate by tr_x and tr_y respectively

    # # Shear
    # pts1 = np.float32([[5,5],[20,5],[5,20]])
    # pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    # pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    # pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    # shear_M = cv2.getAffineTransform(pts1,pts2)

    img_x = cv2.warpAffine(img_x,Rot_M,(cols,rows))
    img_x = cv2.warpAffine(img_x,Trans_M,(cols,rows))
    # img_x = cv2.warpAffine(img_x,shear_M,(cols,rows))
    img_y = cv2.warpAffine(img_y,Rot_M,(cols,rows))
    img_y = cv2.warpAffine(img_y,Trans_M,(cols,rows))
    # img_y = cv2.warpAffine(img_y,shear_M,(cols,rows))

    # Brightness
    if brightness == 1:
        img_x = augment_brightness_camera_images(img_x)

    return img_x, img_y


def augment_img(img_x, img_y):
    """
    genenrate a deformed image for batch image generator, including [rotation, flip, shift]
    because of translation invariance in CNN, i will not do shifting.
    
    """
    #angle = random.randrange(-15, 15)
    #img_x = rotate(img_x, angle, reshape=False)
    #img_y = rotate(img_y, angle, reshape=False)

    img_x, img_y = transform_image(img_x, img_y, 10, 10, 40, brightness=1)
    flip_proba = np.random.random()
    if flip_proba > 0.75: # no flipping
        return img_x, img_y
    elif flip_proba > 0.5: # flip around x-axis
        return cv2.flip(img_x, flipCode=0), cv2.flip(img_y, flipCode=0)
    elif flip_proba > 0.25:# flip around y-axis
        return cv2.flip(img_x, flipCode=1), cv2.flip(img_y, flipCode=1)
    else: # flip around both axes
        return cv2.flip(img_x, flipCode=-1), cv2.flip(img_y, flipCode=-1)



def batch_gen(data_dir, label_df, batch_size=16, is_training=True, image_shape=None, augment=False):
    """
    data_dir:       the directory of JPG images for training
    label_df:       the dataframe with encoded mask strings for each image. The index is image filename
    image_shape:    resize the shape of the image before feeding into the model
    """
    
    # if 'ImageId' in label_df.columns:
    #     label_df = label_df.set_index('ImageId')
    
    img_list = list(label_df.index)
    while True:
        random.shuffle(img_list)
        for start in range(0,len(img_list),batch_size):
            batch_x = []
            batch_y = []
            for img_name in img_list[start:start+batch_size]:
                img_x = cv2.imread(os.path.join(data_dir,img_name))

                if label_df.loc[img_name, 'HasShip'] == 0:
                    if is_training:
                        # skip images without ship during training
                        continue
                    else:
                        img_y = rle_decode_all([])
                else:
                    img_y = rle_decode_all(label_df.loc[img_name,'EncodedPixelsList'])

                # add noise by
                img_x = np.add(img_x, 0.05 * 255 * np.random.randn(*img_x.shape))
                # clip values greater than 255.0, smaller than 0.0 by
                img_x = np.clip(img_x, a_min=0.0, a_max=255.0)

                if image_shape:
                    img_x = cv2.resize(img_x, image_shape)
                    img_y = cv2.resize(img_y, image_shape)
                # img_y = img_y.reshape(img_y.shape+(1,))

                if augment:
                    img_x, img_y = augment_img(img_x, img_y)

                batch_x.append(img_x/255.0)
                batch_y.append(img_y)
            yield np.array(batch_x), np.array(batch_y)





# based on https://www.kaggle.com/kmader/baseline-u-net-model-part-1
def masks_read(data_folder, max_sample = 2000):
    """
    read the true mask file, group by image,
    return a dataframe where mask strings for the same image are combined into a list
    """
    masks = pd.read_csv(os.path.join(data_folder, 'train_ship_segmentations.csv'))
    masks.drop_duplicates(inplace=True)  # drop duplicates in case there is any

    # check whether each instance corresponds to an encoded mask string:
    masks['HasShip'] = masks['EncodedPixels'].map(lambda x: 1 if isinstance(x, str) else 0)
    masks_agg = masks.groupby('ImageId',as_index=False).sum()
    masks_agg['EncodedPixelsList'] = masks.groupby('ImageId',as_index=False)['EncodedPixels'].apply(list)

    # some files are too small/corrupt
    masks_agg['file_size_kb'] = masks_agg['ImageId'].map(lambda img_id:os.stat(os.path.join(data_folder,"train",img_id)).st_size / 1024)
    masks_agg = masks_agg[masks_agg['file_size_kb'] > 50]
    masks_agg.drop('file_size_kb', axis=1, inplace=True)

    # downsample
    df = masks_agg.groupby('HasShip', as_index=False).apply(lambda x: x.sample(max_sample, random_state=99) if len(x) > max_sample else x)
    df.set_index('ImageId', inplace=True)

    assert all(masks_agg['HasShip'].value_counts()[-10:].values == df['HasShip'].value_counts()[-10:].values), "downsampling does not work properly"
    return df.sample(frac=1, random_state=99)





#######################################
#####   Metrics, Loss Functions   #####
#######################################

# check here for the clarification of the metric: https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
def IoU(y_true, y_pred, eps=1e-6):
    # if tf.reduce_max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred)
    TP = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2])
    FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2])
    TN = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=[1,2])
    #intersection = TP
    # union = TP + FP + FN
    iou = (TP + eps)/ (TP + FP + FN + eps)

    # return (TP, FP, FN, TN, iou)
    return tf.reduce_mean(TP), tf.reduce_mean(FP), tf.reduce_mean(FN), tf.reduce_mean(TN), tf.reduce_mean(iou)

def F2_score(y_true, y_pred, beta=2):
    """
    """
    #F2 = tf.zeros(tf.shape(y_pred)[0])
    F2 = []
    for i in range(len(y_pred)):
        pred_masks = label(y_pred[i])
        true_masks = label(y_true[i])
        f2 = []
        for threshold in range(0.5, 1, 0.05):
            tp = 0
            fp = tf.shape(pred_masks)[0]
            fn = tf.shape(true_masks)[0]
            for true_mask in true_masks:
                for pred_mask in pred_masks:
                    iou = IoU(true_mask, pred_mask)[-1]
                    if iou >= threshold:
                        tp += 1
                        fp -= 1
                        fn -= 1
                        break
            f2.append((1 + beta**2) * tp / ((1+beta**2)*tp + (beta**2) * fn + fp))
        F2.append(tf.reduce_mean(f2))
    return tf.reduce_mean(F2)


def BCE_Loss(logits, labels):
    """  binary cross entropy as the loss function, logits are NN predictions before applying activation function  """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def Dice_Loss(logits, labels):
    """  dice loss function, logits are probability prediction, after sigmoid function """
    numerator = 2 * tf.reduce_sum(logits * labels, axis=1)
    denominator = tf.reduce_sum(logits + labels, axis=1)
    return - tf.reduce_mean(tf.log((numerator + 1e-6) / (denominator + 1e-6)))


def Focal_Loss(logits, labels, alpha=0.7, gamma=2):
    """  focal loss function, logits are probability prediction, after sigmoid function """
    # labels = tf.convert_to_tensor(labels)
    # labels = tf.cast(labels, logits.dtype)
    ones = tf.ones_like(labels)
    pt = tf.where(tf.equal(labels, ones), logits, 1.0 - logits)
    tmp = tf.scalar_mul(alpha, ones)
    alpha_t = tf.where(tf.equal(labels, ones), tmp, 1.0 - tmp)
    return - 20. * tf.reduce_mean(alpha_t * tf.pow(1.0 - pt, gamma) * tf.log(pt + 1e-8))
    # define some scaling factor to bring Focal_Loss to be at the same scale as Dice_Loss.
    