from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, filte2):
    return tf.nn.conv2d(x, filte2, strides=[1, 1, 1, 1], padding='SAME')

def conv_layer(x, w_shape, x_shape, b):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    return tf.nn.relu(conv2d(x, w) + b)

def pool_layer(x):
     return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],\
        strides=[1, 2, 2, 1], padding='SAME')

def deconv_layer(x, w_shape, b_shape, name, padding='SAME'):
    w = weight_variable(w_shape)
    b = bias_variable([b_shape])
 
    x_shape = tf.shape(x)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]])
 
    return tf.nn.conv2d_transpose(x, w, out_shape, [1, 1, 1, 1], padding=padding) + b 
'''
 get position and put zeros in
'''
#!Todo: Here should be trace again
def unravel_argmax(self, argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpool_layer(self, x, raveled_argmax, out_shape):
    argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat([t2, t1], 3)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

if __name__ == "__mian__":
    x = tf.placeholder(tf.float32, shape=[None, 288*288])
    x_image = tf.reshape(x, [-1, 288, 288, 1])
    
    conv_1_1 = conv_layer(x_image, [3, 3, 1, 64], 64)
    conv_1_2 = conv_layer(conv_1_1, [3, 3, 64, 64], 64)
    
    pool1, pool1_argmax = pool_layer(conv_1_2)
   
    conv_2_1 = conv_layer(pool1, [3, 3, 64, 128], 128)
    conv_2_2 = conv_layer(conv_2_1, [3, 3, 128, 128], 128)
    
    pool2, pool2_argmax = pool_layer(conv_2_2)
    
    conv_3_1 = conv_layer(pool2, [3, 3, 128, 256], 256)
    conv_3_2 = conv_layer(conv_3_1, [3, 3, 128, 256], 256)
    conv_3_3 = conv_layer(conv_3_2, [3, 3, 128, 256], 256)
    
    pool3, pool3_argmax = pool_layer(conv_3_3)
    
    conv_4_1 = conv_layer(pool3, [3, 3, 256, 512], 512)
    conv_4_2 = conv_layer(conv_4_1, [3, 3, 512, 512], 512)
    conv_4_3 = conv_layer(conv_4_2, [3, 3, 512, 512], 512)

    pool4, pool4_argmax = pool_layer(conv_4_3)

    conv_5_1 = conv_layer(pool4, [3, 3, 256, 512], 512)
    conv_5_2 = conv_layer(conv_5_1, [3, 3, 512, 512], 512)
    conv_5_3 = conv_layer(conv_5_2, [3, 3, 512, 512], 512)

    pool5, pool5_argmax = pool_layer(conv_5_3)

    fc1 = conv_layer(pool5, [9, 9, 512, 4096],4096)
    fc2 = conv_layer(fc1, [1, 1, 4096, 4096],4096)

    deconv_fc2 = deconv_layer(fc2, [9, 9, 512, 4096], 512)

    unpool5 = unpool_layer(deconv_fc2, pool5_argmax, tf.shape(conv_5_3))

    deconv_5_3 = deconv_layer(unpool5, [3, 3, 512, 512], 512)
    deconv_5_2 = deconv_layer(deconv_5_3, [3, 3, 512, 512], 512)
    deconv_5_1 = deconv_layer(deconv_5_2, [3, 3, 512, 512], 512)

    unpool4 = unpool_layer(deconv_5_1, pool4_argmax, tf.shape(conv_4_3))

    deconv_4_3 = deconv_layer(unpool4, [3, 3, 512, 512], 512)
    deconv_4_2 = deconv_layer(deconv_4_3, [3, 3, 512, 512], 512)
    deconv_4_1 = deconv_layer(deconv_4_2, [3, 3, 256, 512], 256)

    unpool3 = unpool_layer(deconv_4_1, pool3_argmax, tf.shape(conv_3_3))

    deconv_3_3 = deconv_layer(unpool3, [3, 3, 256, 256], 256)
    deconv_3_2 = deconv_layer(deconv_3_3, [3, 3, 256, 256], 256)
    deconv_3_1 = deconv_layer(deconv_3_2, [3, 3, 128, 256], 128)

    unpool2 = unpool_layer(deconv_3_1, pool2_argmax, tf.shape(conv_2_2))

    deconv_2_2 = deconv_layer(unpool2, [3, 3, 128, 128], 128)
    deconv_2_1 = deconv_layer(deconv_2_2, [3, 3, 64, 128], 64)

    unpool1 = unpool_layer(deconv_2_1, pool1_argmax, tf.shape(conv_1_2))

    deconv_1_2 = deconv_layer(unpool1, [3, 3, 64, 64], 64)
    deconv_1_1 = deconv_layer(deconv_1_2, [3, 3, 32, 64], 32)

    result = tf.reshape(deconv_1_1, [1, 288, 288, 1])



