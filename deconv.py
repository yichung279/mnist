from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from glob import glob

global DATA
global FILENAME
global BATCH_COUNT
DATA = np.load('npy1/dataEven_1.npy')
FILENAME = 'npy1/dataEven_1.npy'
BATCH_COUNT = 0

def get_next_batch(batch_size, test = False):
    global DATA
    global FILENAME
    global BATCH_COUNT
    
    if not test:
        BATCH_COUNT += 1
        filename = glob('dataEven_*.npy')
        filename.sort()
        # try to do few times of load
        if filename[(BATCH_COUNT / 500) % 6] == FILENAME:
            FILENAME = filename[(BATCH_COUNT / 500) % 6]
            DATA = np.load(FILENAME)
        
        np.random.shuffle(DATA)
    else:
        filename = glob('dataOdd_*.npy')
        DATA = np.load(random.choice(filename))


    label = []
    x = []
    label_holder = [0] * 15


    for i in range(batch_size):
        
        #label
        subLabel = []
        for j in range(DATA.shape[2]):
            label_holder[DATA[i][0][j]] = 1
            subLabel.append(label_holder)
            label_holder[DATA[i][0][j]] = 0
        label.append(subLabel)
        
        #input
        subX = []
        chanel1 = DATA[i][1]
        chanel2 = DATA[i][2]
        chanel3 = DATA[i][3]
        subX.append(chanel1)
        subX.append(chanel2)
        subX.append(chanel3)
        x.append(np.transpose(subX))
        

    label = np.array(label)
    x = np.array(x)


    return [x, label]
    '''
    label = data[:][0]
    chanel1 = data[:][1]
    chanel2 = data[:][2]
    chanel3 = data[:][3]
    '''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, filte2):
    return tf.nn.conv2d(x, filte2, strides=[1, 1, 1, 1], padding='SAME')

def conv_layer(x, w_shape, b_shape):
    w = weight_variable(w_shape)
    b = bias_variable([b_shape])
    return tf.nn.relu(conv2d(x, w) + b)

def pool_layer(x):
     return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],\
        strides=[1, 2, 2, 1], padding='SAME')

def deconv_layer(x, w_shape, b_shape, padding='SAME'):
    w = weight_variable(w_shape)
    b = bias_variable([b_shape])
 
    x_shape = tf.shape(x)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]])
 
    return tf.nn.conv2d_transpose(x, w, out_shape, [1, 1, 1, 1], padding=padding) + b 
'''
 get position and put zeros in
'''
#!Todo: Here should be trace again
def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    
    return tf.stack(output_list)

def unpool_layer(x, raveled_argmax, out_shape):
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
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

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=[None, 288*288, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 288*288, 15])
    x_image = tf.reshape(x, [-1, 288, 288, 3])
    
    conv_1_1 = conv_layer(x_image, [3, 3, 3, 64], 64)
    conv_1_2 = conv_layer(conv_1_1, [3, 3, 64, 64], 64)
    
    pool1, pool1_argmax = pool_layer(conv_1_2)
   
    conv_2_1 = conv_layer(pool1, [3, 3, 64, 128], 128)
    conv_2_2 = conv_layer(conv_2_1, [3, 3, 128, 128], 128)
    
    pool2, pool2_argmax = pool_layer(conv_2_2)
    
    conv_3_1 = conv_layer(pool2, [3, 3, 128, 256], 256)
    conv_3_2 = conv_layer(conv_3_1, [3, 3, 256, 256], 256)
    conv_3_3 = conv_layer(conv_3_2, [3, 3, 256, 256], 256)
    
    pool3, pool3_argmax = pool_layer(conv_3_3)
    
    conv_4_1 = conv_layer(pool3, [3, 3, 256, 512], 512)
    conv_4_2 = conv_layer(conv_4_1, [3, 3, 512, 512], 512)
    conv_4_3 = conv_layer(conv_4_2, [3, 3, 512, 512], 512)

    pool4, pool4_argmax = pool_layer(conv_4_3)

    conv_5_1 = conv_layer(pool4, [3, 3, 512, 512], 512)
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

    score = deconv_layer(deconv_1_1, [1, 1, 15, 32], 15)#0(grayscale)~15
    y_conv = tf.reshape(score, [-1, 15]) 
   
    cross_entropy = tf.reduce_mean(\
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = get_next_batch(100)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={\
                        x: batch[0], y_: batch[1]}) #shape:(batch, 288*288, 3),(batch, 288*288, 15)
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
       
        x_test, y_test = get_next_batch(500, test = True)
        print('test accuracy %g' % accuracy.eval(feed_dict={\
                x: x_test, y_: y_test}))
