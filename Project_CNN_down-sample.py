import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

pic_labels = pd.read_csv('Project_Labels.csv')
week_labels = []
for i in range(len(pic_labels['Week'])):
    week_labels.append(pic_labels['Week'][i])
year_labels = []
for i in range(len(pic_labels['Year'])):
    year_labels.append(pic_labels['Year'][i])
pic_labels = None

input_list = []
for i in range(2409):
    try:
        img = str(i+1)+".jpg"
        im = Image.open(img, 'r')
        im = im.convert(mode="L")
        im = im.resize((160, 160), Image.ANTIALIAS)
        pix_val = list(im.getdata())
        pix_val = np.array(pix_val)
        input_list.append(pix_val)
        print("done", i)
    except FileNotFoundError:
        print("file", i, "not found")
        pass
pix_val = None

input_list = np.array(input_list)
max_of_lists = []
for i in input_list:
    max_of_lists.append(max(i))
input_list = input_list/219

train_input_and_week_labels = [input_list[0:1800], week_labels[0:1800]]
test_input_and_week_labels = [input_list[1800:2408], week_labels[1800:2408]]
train_input_and_year_labels = [input_list[0:1800], year_labels[0:1800]]
test_input_and_year_labels = [input_list[1800:2408], year_labels[1800:2408]]
week_labels = None
year_labels = None
input_list = None


def mini_batch_sampling(ipt, b_size=40):
    new_list1 = []
    new_list2 = []
    idx = []
    for q in range(len(ipt[0])):
        idx.append(q)
    for i in range(int(len(ipt[0])/b_size)):
        batchx = []
        batchy = []
        samp = np.random.choice(idx, size=b_size, replace=False)
        for j in range(b_size):
            batchx.append(ipt[0][samp[j]])
            batchy.append(ipt[1][samp[j]])
            idx.remove(samp[j])
        new_list1.append(batchx)
        new_list2.append(batchy)
    return new_list1, new_list2


batchx = None
batchy = None


def pick_batch(mini_x, mini_y, batch_index):
    return mini_x[batch_index], mini_y[batch_index]


def add_dimension(onedim):
    new_y = []
    for i in onedim:
        if i == 1:
            add = [i, 0]
            new_y.append(np.array(add))
        else:
            add1 = [i, 1]
            new_y.append(np.array(add1))
    new_y = np.array(new_y)
    return new_y


add1 = None


learning_rate = 0.0001
epochs = 10
batch_size = 40

x = tf.placeholder(tf.float32, [None, 25600])
x_shaped = tf.reshape(x, [-1, 160, 160, 1])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+"_W")
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+"_b")
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.leaky_relu(out_layer, alpha=0.05)
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 4, 4, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding="SAME")
    return out_layer


# filter size = 10 x 10, pool size = 4 x 4
layer1 = create_new_conv_layer(x_shaped, 1, 32, [10, 10], [4, 4], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [10, 10], [4, 4], name='layer2')


flattened = tf.reshape(layer2, [-1, 10 * 10 * 64])
flattened = tf.nn.dropout(flattened, keep_prob=keep_prob)
wd1 = tf.Variable(tf.truncated_normal([10*10*64, 500], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([500], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1)+bd1
dense_layer1 = tf.nn.leaky_relu(dense_layer1, alpha=0.05)
dense_layer1 = tf.nn.dropout(dense_layer1, keep_prob=keep_prob)

wd2 = tf.Variable(tf.truncated_normal([500, 2], stddev=0.03), name='wd3')
bd2 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd3')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
dense_layer2 = tf.nn.leaky_relu(dense_layer2, alpha=0.03)
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)
    # number of batches = 45
    total_batches = int(1800/40)
    for epoch in range(epochs):
        avg_cost = 0
        sampling_x, sampling_y = mini_batch_sampling(train_input_and_year_labels, b_size=batch_size)
        for i in range(total_batches):
            batch_x, batch_y = pick_batch(sampling_x, sampling_y, i)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            batch_y = add_dimension(batch_y)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            avg_cost += c/total_batches
            test_year_input = np.array(test_input_and_year_labels[0])
            test_year_labels = np.array(test_input_and_year_labels[1])
            test_year_labels = add_dimension(test_year_labels)
            test_acc = sess.run(accuracy,
                                feed_dict={x: test_year_input, y: test_year_labels, keep_prob: 1})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy:", "{:.3f}".format(test_acc))
    print("\nTraining Complete")
    print(sess.run(accuracy, feed_dict={x: test_year_input, y: test_year_labels, keep_prob: 1})*100, "%")




