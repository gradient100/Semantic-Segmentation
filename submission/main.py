# Vinh Nghiem
# Semantic Segmentation Project
# Udacity Self Driving Car Program

#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

EPOCHS = 50
BATCH_SIZE = 5
LEARNING_RATE = 0.001
KEEP_PROB = 0.50


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    tf.saved_model.loader.load(sess,['vgg16'], vgg_path)
    
    # Get tensors from graph
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')
    
    return image_input, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
    
    # Apply 1x1 convolution in place of fully connected layer
    fcn7 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, padding= 'same',
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # Upsample
    fcn7_up = tf.layers.conv2d_transpose(fcn7, filters=num_classes,
                                         kernel_size=4, strides=(2, 2), padding= 'same',
                                         kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                         kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
                                      
    # Apply 1x1 convolution to 4th layer
    fcn4 = tf.layers.conv2d(layer4, filters=num_classes, kernel_size=1, padding= 'same',
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # Add a skip connection between current final layer fcn7_up and 4th convoluted layer
    fcn7_up_addFcn4 = tf.add(fcn7_up, fcn4)
    
    # Upsample
    fcn7_up_addFcn4_up = tf.layers.conv2d_transpose(fcn7_up_addFcn4, filters=num_classes,
                                                    kernel_size=4, strides=(2, 2), padding= 'same',
                                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
                                   
    # 1x1 convolution to 3rd layer
    fcn3 = tf.layers.conv2d(layer3, filters=num_classes, kernel_size=1, padding= 'same',
                                                           kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                           kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # Add skip connection
    fcn7_up_addFcn4_up_addFcn3 = tf.add(fcn7_up_addFcn4_up, fcn3)
    
    # Upsample
    fcn7_up_addFcn4_up_addFcn3_up = tf.layers.conv2d_transpose(fcn7_up_addFcn4_up_addFcn3, filters=num_classes,
                                                              kernel_size=16, strides=(8, 8), padding= 'same',
                                                              kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                              kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
                                       
    return fcn7_up_addFcn4_up_addFcn3_up
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # make logits 2D so that each row is a pixel and each column is a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # define training loss function
    mce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    # define training operation
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(mce_loss)
    return logits, train_op, mce_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    print ("Training...")
    print()
    for i in range(epochs):
        print ("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:image, correct_label:label, keep_prob:KEEP_PROB, learning_rate:LEARNING_RATE})
            print("Loss: = {:.3f}".format(loss))
            print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_labels = tf.placeholder(tf.int32, [None, None, None, num_classes])
        
        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        fcc_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits, train_op, mce_loss = optimize(fcc_last_layer, correct_labels, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, mce_loss, input_image, correct_labels, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
