"""
MNIST GAN with CNN
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data


X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])


def discriminator(X, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        images = tf.reshape(X, shape=[-1, 28, 28, 1])
        conv_layer_1 = tf.contrib.layers.conv2d(
            inputs=images, num_outputs=32, kernel_size=3, stride=1, padding="SAME",
            activation_fn=tf.nn.relu, trainable=True, scope="conv_layer_1", reuse=reuse,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_layer_2 = tf.contrib.layers.conv2d(
            inputs=conv_layer_1, num_outputs=32, kernel_size=3, stride=1, padding="SAME",
            activation_fn=tf.nn.relu, trainable=True, scope="conv_layer_2", reuse=reuse,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        features = tf.reshape(conv_layer_2, shape=[-1, 28 * 28 * 32])
        fc_layer_1 = tf.contrib.layers.fully_connected(
            inputs=features, num_outputs=128, activation_fn=tf.nn.relu, trainable=True, reuse=reuse,
            weights_initializer=tf.contrib.layers.xavier_initializer(), scope="fc_layer_1")
        prob = tf.contrib.layers.fully_connected(
            inputs=fc_layer_1, num_outputs=1, activation_fn=tf.nn.sigmoid, trainable=True, reuse=reuse,
            weights_initializer=tf.contrib.layers.xavier_initializer(), scope="fc_layer_2")
    return prob


def generator(Z):
    with tf.variable_scope("generator") as scope:
        fc_layer_1 = tf.contrib.layers.fully_connected(
            inputs=Z, num_outputs=128, activation_fn=tf.nn.relu, trainable=True,
            weights_initializer=tf.contrib.layers.xavier_initializer())
        fc_layer_2 = tf.contrib.layers.fully_connected(
            inputs=fc_layer_1, num_outputs=28 * 28 * 32, activation_fn=tf.nn.relu, trainable=True,
            weights_initializer=tf.contrib.layers.xavier_initializer())
        features = tf.reshape(fc_layer_2, shape=[-1, 28, 28, 32])
        conv_layer_1 = tf.contrib.layers.conv2d(
            inputs=features, num_outputs=32, kernel_size=3, stride=1, padding="SAME",
            activation_fn=tf.nn.relu, trainable=True,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_layer_2 = tf.contrib.layers.conv2d(
            inputs=conv_layer_1, num_outputs=1, kernel_size=3, stride=1, padding="SAME",
            activation_fn=tf.nn.sigmoid, trainable=True,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        images = tf.reshape(conv_layer_2, shape=[-1, 784])
    return images


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


with tf.variable_scope("gan") as scope:
    G_sample = generator(Z)
    D_real = discriminator(X)
    D_fake = discriminator(G_sample, reuse=True)
    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))
    D_solver = tf.train.AdamOptimizer().minimize(D_loss,
        var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="gan/discriminator"))
    G_solver = tf.train.AdamOptimizer().minimize(G_loss,
        var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="gan/generator"))


mb_size = 128
Z_dim = 100
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


if not os.path.exists('mnist_gan_conv_out/'):
    os.makedirs('mnist_gan_conv_out/')


i = 0
for it in range(1000000):
    print(it)
    if it % 100 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig('mnist_gan_conv_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    X_mb, _ = mnist.train.next_batch(mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
