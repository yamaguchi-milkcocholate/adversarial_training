import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time
import pickle


def OHE_labels(Y_tr, N_classes):
    OHC = OneHotEncoder()

    Y_ohc = OHC.fit(np.arange(N_classes).reshape(-1, 1))
    Y_labels = Y_ohc.transform(Y_tr.reshape(-1, 1)).toarray()
    return Y_labels


def pre_process_image(image):
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    image = image/255.-.5
    return image


def transform_image(image, ang_range, shear_range, trans_range):
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows, cols, ch = image.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, Rot_M, (cols, rows))
    image = cv2.warpAffine(image, Trans_M, (cols, rows))
    image = cv2.warpAffine(image, shear_M, (cols, rows))
    image = pre_process_image(image)
    return image


def get_index_dict(y_train):
    dict_indices = {}
    ind_all = np.arange(len(y_train))
    for i in range(len(np.unique(y_train))):
        ind_i = ind_all[y_train == i]
        dict_indices[i] = ind_i
    return dict_indices


def gen_transformed_data(X_train, y_train, N_classes, n_each, ang_range, shear_range, trans_range, randomize_Var):
    dict_indices = get_index_dict(y_train)
    n_class = len(np.unique(y_train))
    X_arr = []
    Y_arr = []
    for i in range(n_class):
        len_i = len(dict_indices[i])
        ind_rand = np.random.randint(0,len_i,n_each)
        ind_dict_class = dict_indices[i]

        for i_n in range(n_each):
            img_trf = transform_image(X_train[ind_dict_class[ind_rand[i_n]]],
                                     ang_range, shear_range, trans_range)
            X_arr.append(img_trf)
            Y_arr.append(i)

    X_arr = np.array(X_arr, dtype=np.float32())
    Y_arr = np.array(Y_arr, dtype=np.float32())

    if randomize_Var == 1:
        len_arr = np.arange(len(Y_arr))
        np.random.shuffle(len_arr)
        X_arr[len_arr] = X_arr
        Y_arr[len_arr] = Y_arr

    labels_arr = OHE_labels(Y_arr, 43)

    return X_arr, Y_arr, labels_arr


def gen_extra_data(X_train, y_train, N_classes, n_each, ang_range, shear_range, trans_range, randomize_Var):
    dict_indices = get_index_dict(y_train)
    n_class = len(np.unique(y_train))
    X_arr = []
    Y_arr = []
    n_train = len(X_train)
    for i in range(n_train):
        for i_n in range(n_each):
            img_trf = transform_image(X_train[i], ang_range,shear_range,trans_range)
            X_arr.append(img_trf)
            Y_arr.append(y_train[i])
    X_arr = np.array(X_arr, dtype=np.float32())
    Y_arr = np.array(Y_arr, dtype=np.float32())
    if randomize_Var == 1:
        len_arr = np.arange(len(Y_arr))
        np.random.shuffle(len_arr)
        X_arr[len_arr] = X_arr
        Y_arr[len_arr] = Y_arr
    labels_arr = OHE_labels(Y_arr, 43)
    return X_arr, Y_arr, labels_arr


def random_batch():
    # Number of images in the training-set.
    num_images = len(Image_train_GS_rot_1)
    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)
    # Use the random index to select random images and labels.
    features_batch = Image_train_GS_rot_1[idx, :, :, :]
    labels_batch = labels_train_rot[idx, :]
    return features_batch, labels_batch


def get_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))


def get_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def conv_layer(input,
               num_inp_channels,
               filter_size,
               num_filters,
               use_pooling):
    shape = [filter_size, filter_size, num_inp_channels,num_filters]
    weights = get_weights(shape)
    biases = get_biases(num_filters)
    layer = tf.nn.conv2d(input=input,
                         filters=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(
            input=layer,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def fc_layer(input,          # The previous layer.
             num_inputs,     # Num. inputs from prev. layer.
             num_outputs,    # Num. outputs.
             use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = get_weights(shape=[num_inputs, num_outputs])
    biases = get_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights


def dropout_layer(layer, keep_prob):
    layer_drop = tf.nn.dropout(layer, keep_prob)
    return layer_drop


def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement
    global best_test_accuracy

    global val_acc_list
    global batch_acc_list
    global test_acc_list
    num_iter = num_iterations
    for i in range(num_iter):
        total_iterations += 1
        features_batch, labels_true_batch = random_batch()
        feed_dict_batch = {features: features_batch,
                           labels_true: labels_true_batch,
                           keep_prob: 0.5}
        session.run(optimizer, feed_dict=feed_dict_batch)

        if (total_iterations % 200 == 0) or (i == (num_iter - 1)):
            # Calculate the accuracy on the training-set.
            acc_batch = session.run(accuracy, feed_dict=feed_dict_batch)
            # acc_valid = session.run(accuracy, feed_dict=feed_dict_valid)
            acc_valid = session.run(accuracy, feed_dict=feed_dict_batch)
            val_acc_list.append(acc_valid)
            batch_acc_list.append(acc_batch)
            if acc_valid > best_validation_accuracy:
                best_validation_accuracy = acc_valid
                last_improvement = total_iterations
                improved_str = '*'
                saver = tf.compat.v1.train.Saver()
                saver.save(sess=session, save_path='model_best_batch')
            else:
                improved_str = ''

            if total_iterations - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")
                break

            # Message for printing.
            if (total_iterations % 100 == 0) or (i == (num_iter - 1)):
                msg = "# {0:>6}, Train Acc.: {1:>6.1%}, Val Acc.: {2:>6.1%}, Test Acc.: {3:>6.1%}"
                # acc_test = session.run(accuracy, feed_dict=feed_dict_test)
                acc_test = session.run(accuracy, feed_dict=feed_dict_batch)
                if best_test_accuracy < acc_test:
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess=session, save_path='model_best_test')
                    best_test_accuracy = acc_test
                # Print it.
                print(msg.format(i+1, acc_batch, acc_valid, acc_test))


def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test set: {0:>6.1%}".format(acc))


if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    tf.compat.v1.disable_eager_execution()
    training_file = 'src/storage/datasets/GTSRB/train.p'
    testing_file = 'src/storage/datasets/GTSRB/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    n_class = 43
    labels_train = OHE_labels(y_train, n_class)
    labels_test = OHE_labels(y_test, n_class)
    n_classes = len(np.unique(y_train))

    image_GS_test = np.array([pre_process_image(X_test[i]) for i in range(len(X_test))],
                             dtype=np.float32)
    Image_train_GS_rot, y_train_rot, labels_train_rot = gen_transformed_data(
        X_train, y_train, n_class, 10, 30, 5, 5, 1)
    img_size = 32
    num_channels = 3
    Image_train_GS_rot_1 = Image_train_GS_rot
    image_GS_test_1 = image_GS_test
    features = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='features')
    labels_true = tf.compat.v1.placeholder(tf.float32, shape=[None, n_class], name='y_true')
    labels_true_cls = tf.argmax(labels_true, axis=1)

    ## Convlayer 0
    filter_size0 = 1
    num_filters0 = 3
    ## Convlayer 1
    filter_size1 = 5
    num_filters1 = 32
    ## Convlayer 2
    filter_size2 = 5
    num_filters2 = 32
    ## Convlayer 3
    filter_size3 = 5
    num_filters3 = 64
    ## Convlayer 4
    filter_size4 = 5
    num_filters4 = 64
    ## Convlayer 5
    filter_size5 = 5
    num_filters5 = 128
    ## Convlayer 6
    filter_size6 = 5
    num_filters6 = 128
    ## FC_size
    fc_size1 = 1024
    ## FC_size
    fc_size2 = 1024
    ## Dropout
    # drop_prob = 0.5
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    layer_conv0, weights_conv0 = \
        conv_layer(input=features,
                   num_inp_channels=num_channels,
                   filter_size=filter_size0,
                   num_filters=num_filters0,
                   use_pooling=False)

    layer_conv1, weights_conv1 = \
        conv_layer(input=layer_conv0,
                   num_inp_channels=num_filters0,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=False)
    layer_conv2, weights_conv2 = \
        conv_layer(input=layer_conv1,
                   num_inp_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    layer_conv2_drop = dropout_layer(layer_conv2, keep_prob)

    layer_conv3, weights_conv3 = \
        conv_layer(input=layer_conv2_drop,
                   num_inp_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=False)
    layer_conv4, weights_conv4 = \
        conv_layer(input=layer_conv3,
                   num_inp_channels=num_filters3,
                   filter_size=filter_size4,
                   num_filters=num_filters4,
                   use_pooling=True)
    layer_conv4_drop = dropout_layer(layer_conv4, keep_prob)

    layer_conv5, weights_conv5 = \
        conv_layer(input=layer_conv4_drop,
                   num_inp_channels=num_filters4,
                   filter_size=filter_size5,
                   num_filters=num_filters5,
                   use_pooling=False)
    layer_conv6, weights_conv6 = \
        conv_layer(input=layer_conv5,
                   num_inp_channels=num_filters5,
                   filter_size=filter_size6,
                   num_filters=num_filters6,
                   use_pooling=True)
    layer_conv6_drop = dropout_layer(layer_conv6, keep_prob)

    layer_flat2, num_fc_layers2 = flatten_layer(layer_conv2_drop)
    layer_flat4, num_fc_layers4 = flatten_layer(layer_conv4_drop)
    layer_flat6, num_fc_layers6 = flatten_layer(layer_conv6_drop)

    layer_flat = tf.concat([layer_flat2, layer_flat4, layer_flat6], 1)
    num_fc_layers = num_fc_layers2 + num_fc_layers4 + num_fc_layers6

    fc_layer1, weights_fc1 = fc_layer(layer_flat,  # The previous layer.
                                      num_fc_layers,  # Num. inputs from prev. layer.
                                      fc_size1,  # Num. outputs.
                                      use_relu=True)
    fc_layer1_drop = dropout_layer(fc_layer1, keep_prob)

    fc_layer2, weights_fc2 = fc_layer(fc_layer1_drop,  # The previous layer.
                                      fc_size1,  # Num. inputs from prev. layer.
                                      fc_size2,  # Num. outputs.
                                      use_relu=True)
    fc_layer2_drop = dropout_layer(fc_layer2, keep_prob)

    fc_layer3, weights_fc3 = fc_layer(fc_layer2_drop,  # The previous layer.
                                      fc_size2,  # Num. inputs from prev. layer.
                                      n_classes,  # Num. outputs.
                                      use_relu=False)

    labels_pred = tf.nn.softmax(fc_layer3)
    labels_pred_cls = tf.argmax(labels_pred, axis=1)

    regularizers = (tf.nn.l2_loss(weights_conv0)
                    + tf.nn.l2_loss(weights_conv1) + tf.nn.l2_loss(weights_conv2)
                    + tf.nn.l2_loss(weights_conv3) + tf.nn.l2_loss(weights_conv4)
                    + tf.nn.l2_loss(weights_conv5) + tf.nn.l2_loss(weights_conv6)
                    + tf.nn.l2_loss(weights_fc1) + tf.nn.l2_loss(weights_fc2) +
                    tf.nn.l2_loss(weights_fc3))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer3,
                                                            labels=labels_true)
    cost = tf.reduce_mean(cross_entropy) + 1e-5 * regularizers
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    correct_prediction = tf.equal(labels_pred_cls, labels_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    feed_dict_test = {features: image_GS_test_1,
                      labels_true: labels_test,
                      labels_true_cls: y_test,
                      keep_prob: 1.0}
    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    print_accuracy()

    val_acc_list = []
    batch_acc_list = []
    train_acc_list = []
    batch_size = 512

    start_time = time.time()
    total_iterations = 0
    require_improvement = 10000
    ang_rot = 10
    trans_rot = 2
    shear_rot = 2
    n_opt = 40000
    best_test_accuracy = 0.0

    for i_train in range(1):
        best_validation_accuracy = 0.0
        last_improvement = 0

        # Image_train_GS_rot,y_train_rot,labels_train_rot = gen_transformed_data(X_train,y_train,43,5000,30,5,5,1)

        if i_train > -1:
            ang_rot = 10 * 0.9 ** (i_train)
            trans_rot = 2 * 0.9 ** (i_train)
            shear_rot = 2 * 0.9 ** (i_train)
            require_improvement = 5000
            n_opt = 10000

        X_train_SS, X_valid_SS, y_train_SS, y_valid_SS = \
            train_test_split(X_train,
                             y_train,
                             test_size=0.1,
                             random_state=22)
        labels_valid_SS = OHE_labels(y_valid_SS, 43)
        image_GS_valid = np.array([pre_process_image(X_valid_SS[i]) for i in range(len(X_valid_SS))],
                                  dtype=np.float32)

        feed_dict_valid = {features: image_GS_valid,
                           labels_true: labels_valid_SS,
                           labels_true_cls: y_valid_SS,
                           keep_prob: 1.0}

        Image_train_GS_rot, y_train_rot, labels_train_rot = gen_extra_data(X_train_SS, y_train_SS, 43, 5,
                                                                           ang_rot, trans_rot, shear_rot, 1)
        print('Optimization Loop # ' + str(i_train))
        Image_train_GS_rot_1 = Image_train_GS_rot
        print('train data:', Image_train_GS_rot_1.shape)
        # np.reshape(Image_train_GS_rot,(-1,32,32,1))
        #
        total_parameters = 0
        parameters_string = ""

        for variable in tf.compat.v1.trainable_variables():

            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
            if len(shape) == 1:
                parameters_string += ("%s %d \n" % (variable.name, variable_parameters))
            else:
                parameters_string += ("%s %s=%d \n" % (variable.name, str(shape), variable_parameters))

        print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.compat.v1.trainable_variables()), "{:,}".format(total_parameters)))
        optimize(n_opt)
        # print_accuracy()

    end_time = time.time()

    time_diff = end_time - start_time
