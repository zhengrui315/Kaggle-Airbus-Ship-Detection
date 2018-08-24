import os.path
import tensorflow as tf
from _utils import *
import warnings
# from distutils.version import LooseVersion
# import project_tests as tests
import random
import time
from tqdm import *
import math
from glob import glob
from sklearn.model_selection import train_test_split
import shutil
import argparse
from datetime import datetime
import pickle
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class airbus_model():
    def __init__(self, image_shape=(768, 768), model_dir=None, learning_rate=0.0001, continue_training=False,
                 vgg_path="../vgg/vgg16_weights.npz"):
        self.image_shape = image_shape
        self.model_dir = model_dir
        self.lr = learning_rate
        self.continue_training = continue_training
        self.vgg_path = vgg_path

        self.sess = tf.Session()

        if self.continue_training:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_dir)
        else:
            self.x_holder = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3],
                                           name="x_holder")
            self.y_holder = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1]],
                                           name="y_holder")
            self.final = self.conv_layers()
            self.out = tf.sigmoid(self.final, name="final_out")
            reshaped_logits = tf.reshape(self.final, (-1, self.image_shape[0] * self.image_shape[1]))
            reshaped_labels = tf.reshape(self.y_holder, (-1, self.image_shape[0] * self.image_shape[1]))
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_labels),
                name="cross_entropy")
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="train_op")

            self.pixel_pred = tf.cast(self.out > 0.5, tf.float32, name="pixel_pred")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pixel_pred, self.y_holder), tf.float32),
                                           name="accuracy")
            self.iou = tf.identity(IoU(self.y_holder, self.pixel_pred), name="iou")

            self.sess.run(tf.global_variables_initializer())
            self.load_weights()

            assert all(['new' in v.name for v in tf.trainable_variables()]), "please set trainable variables correctly"

    def conv_layers(self):
        ############   down ##############
        # build vgg
        conv1_1 = tf.layers.conv2d(self.x_holder, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv1_1')
        conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2, 2), strides=2, padding='same', name='vgg_pool1')

        conv2_1 = tf.layers.conv2d(pool1, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv2_1')
        conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2, 2), strides=2, padding='same', name='vgg_pool2')

        conv3_1 = tf.layers.conv2d(pool2, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv3_1')
        conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv3_2')
        conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv3_3')
        pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=(2, 2), strides=2, padding='same', name='vgg_pool3')

        conv4_1 = tf.layers.conv2d(pool3, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv4_1')
        conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv4_2')
        conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv4_3')
        pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=(2, 2), strides=2, padding='same', name='vgg_pool4')

        conv5_1 = tf.layers.conv2d(pool4, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv5_1')
        conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv5_2')
        conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                   trainable=False, name='vgg_conv5_3')
        pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=(2, 2), strides=2, padding='same', name='vgg_pool5')

        #############  up  ###################
        # https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
        layer6_1 = tf.layers.conv2d_transpose(pool5, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                              bias_initializer=tf.constant_initializer(0.1),
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              activation=tf.nn.relu, trainable=True, name='new_layer6_1')
        layer6_2 = tf.concat([pool4, layer6_1], axis=-1, name='new_layer6_2')

        layer7_1 = tf.layers.conv2d_transpose(layer6_2, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                              bias_initializer=tf.constant_initializer(0.1),
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              activation=tf.nn.relu, trainable=True, name='new_layer7_1')
        layer7_2 = tf.concat([pool3, layer7_1], axis=-1, name='new_layer7_2')

        layer8_1 = tf.layers.conv2d_transpose(layer7_2, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                              bias_initializer=tf.constant_initializer(0.1),
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              activation=tf.nn.relu, trainable=True, name='new_layer8_1')
        layer8_2 = tf.concat([pool2, layer8_1], axis=-1, name='new_layer8_2')

        layer9_1 = tf.layers.conv2d_transpose(layer8_2, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                              bias_initializer=tf.constant_initializer(0.1),
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              activation=tf.nn.relu, trainable=True, name='new_layer9_1')
        layer9_2 = tf.concat([pool1, layer9_1], axis=-1, name='new_layer9_2')

        # the last layer should be linear, no activation function
        logits = tf.layers.conv2d_transpose(layer9_2, filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            bias_initializer=tf.constant_initializer(0.1),
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            activation=None, trainable=True, name='new_logits')
        return tf.squeeze(logits, axis=-1)  # remove the channel dimension which has size 1

    def load_weights(self):
        weights = np.load(self.vgg_path)
        d = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if var.name.startswith('vgg_')}
        # print(d.keys())
        for new_name in d.keys():
            # print(new_name)
            # observe d.keys() and weights.keys() to see the difference in naming
            if new_name.endswith('/kernel:0'):
                old_name = new_name[4:11] + '_W'
            elif new_name.endswith('/bias:0'):
                old_name = new_name[4:11] + '_b'
            self.sess.run(d[new_name].assign(weights[old_name]))
            # print(old_name)

    def train(self, epochs, data_folder, label_df):

        train_ids, valid_ids = train_test_split(label_df.index.values, train_size=0.8,
                                                stratify=label_df['HasShip'].values)
        train_df = label_df.loc[train_ids]
        valid_df = label_df.loc[valid_ids]

        batch_size = 16
        # Create function to get batches
        training_batch_generator = batch_gen(data_folder, train_df, batch_size, image_shape=self.image_shape,
                                             augment=True)

        samples_per_epoch = len(train_ids)
        batches_per_epoch = samples_per_epoch // batch_size

        training_loss_metrics = []
        training_accuracy_metrics = []
        training_iou_metrics = []
        validation_loss_metrics = []
        validation_accuracy_metrics = []
        validation_iou_metrics = []

        print("evaluating validation...")
        validation_loss, validation_accuracy, validation_iou = self.evaluate(data_folder, valid_df)
        print("Validation loss: %.4f, accuracy: %.5f, iou: %.3f" % (validation_loss, validation_accuracy, validation_iou))

        for epoch in range(epochs):
            print("Epochs {} ... \n".format(epoch + 1))
            for _ in tqdm(range(batches_per_epoch)):
                X_batch, y_batch = next(training_batch_generator)
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.x_holder: X_batch,
                    self.y_holder: y_batch})

            print("Evaluating validation...")
            validation_loss, validation_accuracy, validation_iou = self.evaluate(data_folder, valid_df)
            validation_loss_metrics.append(validation_loss)
            validation_accuracy_metrics.append(validation_accuracy)
            validation_iou_metrics.append(validation_iou)
            print("Validation loss: %.4f, accuracy: %.5f, iou: %.3f" % (
            validation_loss, validation_accuracy, validation_iou))

            #print("Evaluating training...")
            training_loss, training_accuracy, training_iou = (1, 1, 1)  # self.evaluate(data_folder, train_df)
            training_loss_metrics.append(training_loss)
            training_accuracy_metrics.append(training_accuracy)
            training_iou_metrics.append(training_iou)
            #print("Training loss: %.4f, accuracy: %.5f, iou: %.3f" % (training_loss, training_accuracy, training_iou))
            # self.debug2(data_folder, label_df)
        print("loss, acc, iou: ")
        print(validation_loss_metrics)
        print(validation_accuracy_metrics)
        print(validation_iou_metrics)
        return (training_loss_metrics, training_accuracy_metrics, training_iou_metrics, validation_loss_metrics,
                validation_accuracy_metrics, validation_iou_metrics)

    def evaluate(self, data_folder, label_df):
        batch_size = 16
        data_generator = batch_gen(data_folder, label_df, batch_size, image_shape=self.image_shape, augment=False)
        num_examples = (label_df.shape[0] // batch_size) * batch_size
        total_loss = 0
        total_acc = 0
        total_iou = 0
        for _ in tqdm(range(0, num_examples, batch_size)):
            X_batch, y_batch = next(data_generator)
            loss, accuracy, iou = self.sess.run([self.loss, self.accuracy, self.iou],
                                                feed_dict={self.x_holder: X_batch, self.y_holder: y_batch})
            total_loss += loss
            total_acc += accuracy
            total_iou += iou
        return total_loss / (num_examples / batch_size), total_acc / (num_examples / batch_size), total_iou / (
        num_examples / batch_size)

    def debug1(self, data_folder, label_df):
        """ use the 3rd image to check whether the code is working properly """
        print("checking image 3...")
        img_name = label_df.index[2]
        x = cv2.imread(os.path.join(data_folder, img_name)) / 255.0
        if label_df.loc[img_name, 'HasShip'] == 0:
            y = rle_decode_all([])
        else:
            y = rle_decode_all(label_df.loc[img_name, 'EncodedPixelsList'])

        print("np.sum(y)=", np.sum(y))

        x = x[np.newaxis, :]
        y = y.reshape((1,) + y.shape)
        out = self.sess.run(self.out, feed_dict={self.x_holder: x})
        out = tf.squeeze(out)
        out = tf.cast(out > 0.5, tf.float32)
        out = self.sess.run(out)
        print("np.sum(out)=", np.sum(out))

        pred = out == np.squeeze(y)
        pred = pd.Series(pred.reshape((-1)))
        c = pred.value_counts(normalize=True)
        acc = self.sess.run(self.accuracy, feed_dict={self.x_holder: x, self.y_holder: y})
        print("accuracy for image 3:", acc)
        assert round(acc, 4) == round(c[True].astype(np.float32), 4), " something is wrong with the code"
        print()

    def debug2(self, data_folder, label_df):
        """ use the 3rd image to check whether the code is working properly """
        print("checking image 3...")
        img_name = label_df.index[2]
        x = cv2.imread(os.path.join(data_folder, img_name)) / 255.0
        if label_df.loc[img_name, 'HasShip'] == 0:
            y = rle_decode_all([])
        else:
            y = rle_decode_all(label_df.loc[img_name, 'EncodedPixelsList'])

        print("np.sum(y)=", np.sum(y))

        x = x.reshape((1,) + x.shape)
        y_true = y.reshape((1,) + y.shape)
        print("\n SHAPE = ", x.shape, y_true.shape, end="\n\n")
        y_pred, iou = self.sess.run([self.pixel_pred, self.iou], feed_dict={self.x_holder: x, self.y_holder: y_true})
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        eps = 1e-6
        iou2 = -(intersection + eps) / (union + eps)
        print("iou1, iou2 = ", iou, iou2)
        # assert iou == iou2, "something is wrong with the definition of IoU function!!!"

    def save(self, metrics):
        print("Saving the model ...")

        if not self.model_dir:
            self.model_dir = "./save_model"
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(self.model_dir, "airbus_model"))

        train_loss, train_accuracy, train_iou, valid_loss, valid_accuracy, valid_iou = metrics
        # metrics_list = ("train_loss", "train_accuracy", "train_iou", "valid_loss", "valid_accuracy", "valid_iou")

        if self.continue_training:
            #    for metric in metrics_list:
            #        with open(os.path.join(self.model_dir, metric), "rb") as f:

            with open(os.path.join(self.model_dir, "train_loss"), "rb") as f:
                train_loss = pickle.load(f) + train_loss
            with open(os.path.join(self.model_dir, "train_accuracy"), "rb") as f:
                train_accuracy = pickle.load(f) + train_accuracy
            with open(os.path.join(self.model_dir, "train_iou"), "rb") as f:
                train_iou = pickle.load(f) + train_iou

            with open(os.path.join(self.model_dir, "valid_loss"), "rb") as f:
                valid_loss = pickle.load(f) + valid_loss
            with open(os.path.join(self.model_dir, "valid_accuracy"), "rb") as f:
                valid_accuracy = pickle.load(f) + valid_accuracy
            with open(os.path.join(self.model_dir, "valid_iou"), "rb") as f:
                valid_iou = pickle.load(f) + valid_iou

        with open(os.path.join(self.model_dir, "train_loss"), 'wb') as f:
            pickle.dump(train_loss, f)
        with open(os.path.join(self.model_dir, "train_accuracy"), 'wb') as f:
            pickle.dump(train_accuracy, f)
        with open(os.path.join(self.model_dir, "train_iou"), 'wb') as f:
            pickle.dump(train_iou, f)

        with open(os.path.join(self.model_dir, "valid_loss"), 'wb') as f:
            pickle.dump(valid_loss, f)
        with open(os.path.join(self.model_dir, "valid_accuracy"), "wb") as f:
            pickle.dump(valid_accuracy, f)
        with open(os.path.join(self.model_dir, "valid_iou"), 'wb') as f:
            pickle.dump(valid_iou, f)


def main():

    parser = argparse.ArgumentParser(description='Airbus Nails it')
    parser.add_argument(
        '-n',
        '--num_epochs',
        type=int,
        nargs='?',
        default=1,
        help='Number of epochs.'
    )
    parser.add_argument("-ct", "--continues_training", help="Continue from where you left off", action="store_true")
    args = parser.parse_args()
    num_epochs = args.num_epochs
    CONTINUE_TRAINING = args.continues_training

    #image_shape = (224, 224)
    data_folder = os.path.join(os.getcwd(), '../data/train')
    label_df = masks_read(os.path.join(os.getcwd(), '../data'))
    if CONTINUE_TRAINING:
        model = airbus_model(model_dir=os.path.join(os.getcwd(), 'save_model'))
    else:
        model = airbus_model()
    #model.debug2(data_folder, label_df)
    metrics = model.train(num_epochs, data_folder, label_df)

    model.save(metrics)


if __name__ == '__main__':
    main()