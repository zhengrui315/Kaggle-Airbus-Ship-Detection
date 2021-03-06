import os.path
import tensorflow as tf
from _utils import *
from tqdm import *
from sklearn.model_selection import train_test_split
import pickle
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class airbus_base():
    def __init__(self, model_dir, image_shape=(768, 768), batch_size=4, debug=1, continue_training=False, learning_rate=1e-4):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.debug = debug
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self.lr = learning_rate
        self.continue_training = continue_training

        self.sess = tf.Session()
        #tf.set_random_seed(918)

        self.x_holder = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3],
                                       name="x_holder")
        self.y_holder = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1]],
                                       name="y_holder")
        self.final = self.conv_layers()
        self.out = tf.sigmoid(self.final, name="final_out")

        reshaped_logits = tf.reshape(self.out, (-1, self.image_shape[0] * self.image_shape[1]))
        reshaped_labels = tf.reshape(self.y_holder, (-1, self.image_shape[0] * self.image_shape[1]))

        self.bce_loss = tf.identity(BCE_Loss(self.final, self.y_holder), name="bce_loss")
        self.focal_loss = tf.identity(Focal_Loss(reshaped_logits, reshaped_labels), name="focal_loss")
        self.dice_loss = tf.zeros(1) #tf.identity(Dice_Loss(reshaped_logits, reshaped_labels), name="dice_loss")
        self.loss = tf.add(self.focal_loss, self.dice_loss, name="total_loss")

        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decay_steps = 30
        decay_rate = 0.1
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(self.lr, global_step, decay_steps, decay_rate, staircase=False)

        down_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='downsample')
        up_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='upsample')
        trainable_variables = up_variables + down_variables
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.bce_loss, var_list = trainable_variables, name="train_op")

        self.pixel_pred = tf.cast(self.out > 0.5, tf.float32, name="pixel_pred")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pixel_pred, self.y_holder), tf.float32),
                                       name="accuracy")
        self.iou = tf.identity(IoU(self.y_holder, self.pixel_pred), name="iou")
        self.f2 = tf.identity(F2_score(self.y_holder, self.pixel_pred), name="f2")

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
            #print("There are {} trainable parameters in the layers".format(self.sess.run(param_count)))
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


        # Create function to get batches
        training_batch_generator = batch_gen(data_folder, train_df, self.batch_size, image_shape=self.image_shape,
                                             augment=True)

        samples_per_epoch = len(train_ids)
        batches_per_epoch = samples_per_epoch // self.batch_size

        train_loss_metrics = []
        train_acc_metrics = []
        train_iou_metrics = []
        val_loss_metrics = []
        val_acc_metrics = []
        val_iou_metrics = []

        # early stopping initialization
        if self.continue_training == False:
            best_val_loss = np.infty
            check_progress_num = 0
        else:
            with open(os.path.join("./save_model", "valid_loss"), "rb") as f:
                val_loss_his = pickle.load(f)
            best_val_loss = min(val_loss_his)
            check_progress_num = len(val_loss_his) - 1 - val_loss_his.index(best_val_loss)
        max_check_progress_num = 5

        print("Evaluating validation in the beginning ...")
        val_loss, val_acc, val_iou, val_f2 = self.evaluate(data_folder, valid_df)
        print("Validation loss: %.6f, %.6f, %.6f" % (val_loss[0], val_loss[1], val_loss[2]))
        print("Valication accuracy: %.5f" % val_acc)
        print("Valication iou:", val_iou)
        #print("Valication F2:", val_f2)

        for epoch in range(epochs):
            print()
            print("Epochs {} ... ".format(epoch + 1))
            for _ in tqdm(range(batches_per_epoch//self.debug)):
                X_batch, y_batch = next(training_batch_generator)
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.x_holder: X_batch,
                    self.y_holder: y_batch})

            print("Evaluating validation...")
            val_loss, val_acc, val_iou, val_f2 = self.evaluate(data_folder, valid_df)
            val_loss_metrics.append(val_loss[0])
            val_acc_metrics.append(val_acc)
            val_iou_metrics.append(val_iou[4])
            print("Validation loss: %.6f, %.6f, %.6f" % (val_loss[0],val_loss[1],val_loss[2]))
            print("Valication accuracy: %.5f" % val_acc)
            print("Valication iou:", val_iou)
            #print("Valication F2 score:", val_f2)

            print("Evaluating training...")
            train_loss, train_acc, train_iou, train_f2 = self.evaluate(data_folder, train_df.sample(n=80))
            train_loss_metrics.append(train_loss[0])
            train_acc_metrics.append(train_acc)
            train_iou_metrics.append(train_iou[4])
            print("Training loss: %.6f, %.6f, %.6f" % (train_loss[0],train_loss[1],train_loss[2]))
            print("Training accuracy: %.5f" % train_acc)
            print("Training iou:", train_iou)
            #print("Training F2 score:", train_f2)

            if val_loss[0] < best_val_loss:
                if self.debug == 1:
                    print("saving checkpoint at epoch {} ...".format(epoch+1))
                    self.saver.save(self.sess, os.path.join(self.model_dir, "airbus_model"))
                best_val_loss = val_loss[0]
                check_progress_num = 0
            else:
                check_progress_num += 1
                if check_progress_num > max_check_progress_num:
                    print("Early Stopping at Epoch {}".format(epoch+1))
                    break

        print("loss, acc, iou: ")
        print(val_loss_metrics)
        print(val_acc_metrics)
        print(val_iou_metrics)

        if self.debug == 1:
            self.save_history((train_loss_metrics, train_acc_metrics, train_iou_metrics, val_loss_metrics, val_acc_metrics, val_iou_metrics))

        print("DONE!")


    def evaluate(self, data_folder, label_df):
        data_generator = batch_gen(data_folder, label_df, self.batch_size, is_training=False, image_shape=self.image_shape, augment=False)
        num_examples = (label_df.shape[0] // self.batch_size) * self.batch_size

        loss = [self.loss, self.focal_loss, self.dice_loss]
        total_loss = [0,0,0] # total_loss, total_focal_loss, total_dice_loss
        total_acc = 0
        total_iou = [0,0,0,0,0] #  acc, TP, FP, FN, TN, iou
        total_f2 = 0
        for _ in tqdm(range(0, num_examples//self.debug, self.batch_size)):
            X_batch, y_batch = next(data_generator)
            results = self.sess.run([loss, self.accuracy, self.iou, self.f2], feed_dict={self.x_holder: X_batch, self.y_holder: y_batch})
            total_loss = [total_loss[i] + results[0][i] for i in range(len(results[0]))]
            total_acc += results[1]
            total_iou = [total_iou[i] + results[2][i] for i in range(len(results[2]))]
            total_f2 += results[3]
        return [x / (num_examples / self.batch_size) for x in total_loss], total_acc / (num_examples / self.batch_size), [x / (num_examples / self.batch_size) for x in total_iou], total_f2 / (num_examples / self.batch_size)


    def save_history(self, metrics):
        print("Saving History ...")

        train_loss, train_accuracy, train_iou, valid_loss, valid_accuracy, valid_iou = metrics

        if self.continue_training and os.path.isfile(os.path.join(self.model_dir, "train_loss")):

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




class VGG16(airbus_base):
    def __init__(self, vgg_path = "../vgg/vgg16_weights.npz", **kwargs):
        self.vgg_path = vgg_path
        super(VGG16, self).__init__(**kwargs)

    def conv_layers(self):
        ############   down ##############
        # build vgg
        with tf.variable_scope('downsample'):
            pool1 = self.vgg_block(input=self.x_holder, filters=64, num_layers=2, block_id=1)
            pool2 = self.vgg_block(input=pool1, filters=128, num_layers=2, block_id=2)
            pool3 = self.vgg_block(input=pool2, filters=256, num_layers=3, block_id=3)
            pool4 = self.vgg_block(input=pool3, filters=512, num_layers=3, block_id=4)
            pool5 = self.vgg_block(input=pool4, filters=512, num_layers=3, block_id=5)

        #tf.stop_gradient(pool5)

        #############  up  ##################
        down_inputs = [pool5, pool4, pool3, pool2, pool1]
        logits = self.upsampling(down_inputs)

        return tf.squeeze(logits, axis=-1)  # remove the channel dimension which has size 1

    def vgg_block(self, input, filters, num_layers, block_id):
        conv = input
        for i in range(num_layers):
            conv = tf.layers.conv2d(conv, filters=filters, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='vgg_conv'+str(block_id)+'_'+str(i+1))
        conv = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=2, padding='same', name='vgg_pool'+str(block_id))
        return conv

    def load_weights0(self):
        weights = np.load(self.vgg_path)
        d = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if var.name.startswith('vgg_')}
        # print(d.keys())
        for new_name in d.keys():
            # print(new_name)
            # observe d.keys() and weights.keys() to see the difference in naming
            if new_name.endswith('/kernel:0'):
                old_name = re.findall(r'conv\d*_\d', new_name)[0] + '_W'
            elif new_name.endswith('/bias:0'):
                old_name = re.findall(r'conv\d*_\d', new_name)[0] + '_b'
            self.sess.run(d[new_name].assign(weights[old_name]))
            # print(old_name)

    def load_weights(self):
        print("Loading new weights ...")
        with open("./weights", "rb") as f:
            weights = pickle.load(f)
        d = {var.name.strip('downupsample').strip('/'): var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if var.name != 'global_step:0' and 'Adam' not in var.name}
        for var in d.keys():
            self.sess.run(d[var].assign(weights[var]))


    def upsampling(self, down_inputs):
        raise NotImplementedError(" upsampling layers need to be implemented")