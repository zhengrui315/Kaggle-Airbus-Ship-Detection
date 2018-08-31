import os.path
import tensorflow as tf
from _utils import *
from tqdm import *
from sklearn.model_selection import train_test_split
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class airbus_base():
    def __init__(self, image_shape, model_dir, continue_training=False, learning_rate=0.005):
        self.image_shape = image_shape
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self.lr = learning_rate
        self.continue_training = continue_training

        self.sess = tf.Session()

        self.x_holder = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3],
                                       name="x_holder")
        self.y_holder = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1]],
                                       name="y_holder")
        self.final = self.conv_layers()
        self.out = tf.sigmoid(self.final, name="final_out")

        reshaped_logits = tf.reshape(self.final, (-1, self.image_shape[0] * self.image_shape[1]))
        reshaped_labels = tf.reshape(self.y_holder, (-1, self.image_shape[0] * self.image_shape[1]))

        # https://www.jeremyjordan.me/semantic-segmentation/#fully_convolutional
        self.bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_labels), name="bce_loss")
        numerator = 2 * tf.reduce_sum(reshaped_logits * reshaped_labels, axis=1)
        denominator = tf.reduce_sum(reshaped_logits + reshaped_labels, axis=1)
        self.dice_loss = 1 - tf.reduce_mean(numerator / denominator, name="dice_loss")
        self.loss = tf.add(self.bce_loss, self.dice_loss, name="cross_entropy")

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="train_op")

        self.pixel_pred = tf.cast(self.out > 0.5, tf.float32, name="pixel_pred")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pixel_pred, self.y_holder), tf.float32),
                                       name="accuracy")
        self.iou = tf.identity(IoU(self.y_holder, self.pixel_pred), name="iou")

        if self.continue_training:
            print("Continue Training ...")
            restore_saver = tf.train.Saver()
            restore_saver.restore(self.sess, os.path.join(self.model_dir, "airbus_model"))
        else:
            self.sess.run(tf.global_variables_initializer())
            self.load_weights()

            # check trainable variables:
            # assert all(['new' in v.name for v in tf.trainable_variables()]), "please set trainable variables correctly"
            param_count = [tf.reduce_prod(v.shape) for v in tf.trainable_variables()]
            print("There are {} trainable parameters in the layers".format(self.sess.run(param_count)))
            print("There are {} trainable parameters".format(self.sess.run(tf.reduce_sum(param_count))))

        self.saver = tf.train.Saver()

    def conv_layers(self):
        raise NotImplementedError

    def load_weights(self):
        pass

    def train(self, epochs, data_folder, label_df):

        train_ids, valid_ids = train_test_split(label_df.index.values, train_size=0.8,
                                                stratify=label_df['HasShip'].values, random_state=99)
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

        # early stopping initialization
        if self.continue_training == False:
            best_validation_loss = np.infty
            check_progress_num = 0
        else:
            with open(os.path.join("./save_model", "valid_loss"), "rb") as f:
                val_loss_his = pickle.load(f)
            best_validation_loss = min(val_loss_his)
            check_progress_num = len(val_loss_his) - 1 - val_loss_his.index(best_validation_loss)
        max_check_progress_num = 5

        print("Evaluating validation in the beginning ...")
        validation_loss, validation_accuracy, validation_iou = self.evaluate(data_folder, valid_df)
        print("Validation loss: %.6f, accuracy: %.5f, iou: %.3f" % (validation_loss, validation_accuracy, validation_iou[-1]))

        for epoch in range(epochs):
            print("Epochs {} ... ".format(epoch + 1))
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
            print("Validation loss: %.6f, accuracy: %.5f, iou: %.5f" % (validation_loss, validation_accuracy, validation_iou[-1]))
            print("iou:", validation_iou)

            print("Evaluating training...")
            training_loss, training_accuracy, training_iou = validation_loss, validation_accuracy, validation_iou# self.evaluate(data_folder, train_df)
            training_loss_metrics.append(training_loss)
            training_accuracy_metrics.append(training_accuracy)
            training_iou_metrics.append(training_iou)
            print("Training loss: %.6f, accuracy: %.5f, iou: %.3f" % (training_loss, training_accuracy, training_iou[-1]))
            # self.debug2(data_folder, label_df)

            if validation_loss < best_validation_loss:
                print("saving checkpoint at epoch {} ...".format(epoch+1))
                #self.saver.save(self.sess, os.path.join(self.model_dir, "airbus_model"))
                best_validation_loss = validation_loss
                check_progress_num = 0
            else:
                check_progress_num += 1
                if check_progress_num > max_check_progress_num:
                    print("Early Stopping at Epoch {}".format(epoch+1))
                    break

        print("loss, acc, iou: ")
        print(validation_loss_metrics)
        print(validation_accuracy_metrics)
        print(validation_iou_metrics)

        print("Saving History ...")
        #self.save_history((training_loss_metrics, training_accuracy_metrics, training_iou_metrics, validation_loss_metrics,
        #        validation_accuracy_metrics, validation_iou_metrics))

    def evaluate(self, data_folder, label_df):
        batch_size = 16
        data_generator = batch_gen(data_folder, label_df, batch_size, is_training=False, image_shape=self.image_shape, augment=False)
        num_examples = (label_df.shape[0] // batch_size) * batch_size
        total_loss = 0
        total_acc = 0
        total_iou = [0,0,0,0,0]
        for _ in tqdm(range(0, num_examples, batch_size)):
            X_batch, y_batch = next(data_generator)
            loss, accuracy, iou = self.sess.run([self.loss, self.accuracy, self.iou],
                                                feed_dict={self.x_holder: X_batch, self.y_holder: y_batch})
            total_loss += loss
            total_acc += accuracy
            total_iou = [total_iou[i] + iou[i] for i in range(len(iou))]
        return total_loss / (num_examples / batch_size), total_acc / (num_examples / batch_size), [x / (num_examples / batch_size) for x in total_iou]


    def save_history(self, metrics):
        print("Saving the history ...")

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




class airbus_scratch(airbus_base):
    def __init__(self, image_shape = (768, 768), model_dir = "./save_model", continue_training = False):
        super(airbus_scratch,self).__init__(image_shape=image_shape, model_dir=model_dir, continue_training=continue_training)

    def conv_layers(self):
        # downsample
        conv1_1 = tf.layers.conv2d(self.x_holder, filters=32, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv1_1')
        conv1_2 = tf.layers.conv2d(conv1_1, filters=32, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2,2), strides=2, padding='same', name='pool1')

        conv2_1 = tf.layers.conv2d(pool1, filters=32, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv2_1')
        conv2_2 = tf.layers.conv2d(conv2_1, filters=32, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2,2), strides=2, padding='same', name='pool2')


        conv3_1 = tf.layers.conv2d(pool2, filters=32, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv3_1')
        conv3_2 = tf.layers.conv2d(conv3_1, filters=32, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv3_2')
        pool3 = tf.layers.max_pooling2d(conv3_2, pool_size=(2,2), strides=2, padding='same', name='pool3')


        conv4_1 = tf.layers.conv2d(pool3, filters=64, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv4_1')
        conv4_2 = tf.layers.conv2d(conv4_1, filters=64, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv4_2')
        pool4 = tf.layers.max_pooling2d(conv4_2, pool_size=(2,2), strides=2, padding='same', name='pool4')


        conv5_1 = tf.layers.conv2d(pool4, filters=64, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv5_1')
        conv5_2 = tf.layers.conv2d(conv5_1, filters=64, kernel_size=(3,3), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv5_2')
        pool5 = tf.layers.max_pooling2d(conv5_2, pool_size=(2,2), strides=2, padding='same', name='pool5')


        # upsample
        conv6 = tf.layers.conv2d_transpose(pool5, filters=64, kernel_size=(3,3), strides=(2,2), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv6')
        add46 = tf.concat([conv6, pool4], axis=-1, name='add_4_6')

        conv7 = tf.layers.conv2d_transpose(add46, filters=32, kernel_size=(3,3), strides=(2,2), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv7')
        add73 = tf.concat([conv7, pool3], axis=-1, name='add_3_7')

        conv8 = tf.layers.conv2d_transpose(add73, filters=32, kernel_size=(3,3), strides=(2,2), padding='same',
                                         bias_initializer=tf.constant_initializer(0.1),
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv8')
        add82 = tf.add(conv8, pool2, name='add_2_8')

        conv9 = tf.layers.conv2d_transpose(add82, filters=32, kernel_size=(3,3), strides=(2,2), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.nn.relu, trainable=True, name='conv9')
        add91 = tf.concat([conv9, pool1], axis=-1, name='add_1_9')

        # the last layer should be linear, so no activation function
        logits = tf.layers.conv2d_transpose(add91, filters=1, kernel_size=(3,3), strides=(2,2), padding='same',
                                          bias_initializer=tf.constant_initializer(0.1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=None, trainable=True, name='logits')
        return tf.squeeze(logits, axis=-1) # remove the channel dimension which has size 1


class airbus_vgg(airbus_base):
    def __init__(self, image_shape = (768, 768), model_dir = "./save_model", continue_training = False, vgg_path = "../vgg/vgg16_weights.npz"):
        self.vgg_path = vgg_path
        super(airbus_vgg,self).__init__(image_shape=image_shape, model_dir=model_dir, continue_training=continue_training)

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

