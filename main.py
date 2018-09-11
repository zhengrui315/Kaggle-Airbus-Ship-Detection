from model import airbus_scratch, VGG16
from _utils import *
import os
import tensorflow as tf
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class airbus_vgg(VGG16):
    def __init__(self, **kwargs):
        super(airbus_vgg, self).__init__(**kwargs)

    def upsampling(self, down_inputs):
        with tf.variable_scope('upsample'):
            layer6 = self.up_block([down_inputs[1], down_inputs[0]], 512, 6)
            layer7 = self.up_block([down_inputs[2], layer6], 256, 7)
            layer8 = self.up_block([down_inputs[3], layer7], 128, 8)
            layer9 = self.up_block([down_inputs[4], layer8], 64, 9)
            # the last layer should be linear, no activation function
            layer10 = tf.layers.conv2d_transpose(layer9, filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                bias_initializer=tf.zeros_initializer, #tf.constant_initializer(0.1),
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                activation=tf.nn.relu, trainable=True, name='new_layer_10')
            logits = tf.layers.conv2d(layer10, filters=1, kernel_size=(3, 3), padding='same',
                             bias_initializer=tf.zeros_initializer,  # tf.constant_initializer(0.1),
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                             activation=None, trainable=True, name='new_logits')
        return logits


    def up_block(self, inputs, filters, num):
        """  inputs = [down_input, up_input]  """
        layer_1 = tf.layers.conv2d_transpose(inputs[1], filters=filters // 2, kernel_size=(3, 3), strides=(2, 2),
                                             padding='same',
                                             bias_initializer=tf.zeros_initializer, #tf.constant_initializer(0.1),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                             activation=tf.nn.relu, trainable=True, name='new_layer' + str(num) + '_1')
        layer_2 = tf.concat([inputs[0], layer_1], axis=-1, name='new_layer' + str(num) + '_2')
        layer_3 = tf.layers.conv2d(layer_2, filters=filters, kernel_size=(3, 3), padding='same',
                                   bias_initializer=tf.zeros_initializer, #tf.constant_initializer(0.1),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                   activation=tf.nn.relu, trainable=True, name='new_layer' + str(num) + '_3')
        layer_4 = tf.layers.conv2d(layer_3, filters=filters, kernel_size=(3, 3), padding='same',
                                   bias_initializer=tf.zeros_initializer, #tf.constant_initializer(0.1),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                   activation=tf.nn.relu, trainable=True, name='new_layer' + str(num) + '_4')
        return layer_4




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
    NUM_EPOCHS = args.num_epochs
    CONTINUE_TRAINING = args.continues_training

    #image_shape = (224, 224)
    data_folder = os.path.join(os.getcwd(), '../data/train')
    model_folder = os.path.join(os.getcwd(), 'save_model')
    label_df = masks_read(os.path.join(os.getcwd(), '../data'))

    airbus_model = airbus_vgg(model_dir=model_folder, continue_training=CONTINUE_TRAINING)

    airbus_model.train(NUM_EPOCHS, data_folder, label_df)


if __name__ == '__main__':
    main()