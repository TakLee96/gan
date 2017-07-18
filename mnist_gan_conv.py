"""
MNIST GAN with CNN
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data


mb_size = 128
Z_dim = 100


X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])


def discriminator(X, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        fc_layer_1 = tf.contrib.layers.fully_connected(
            inputs=X, num_outputs=128, activation_fn=tf.nn.relu, trainable=True,
            scope="fc_layer_1", reuse=reuse,
            weights_initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.contrib.layers.fully_connected(
            inputs=fc_layer_1, num_outputs=1, activation_fn=None, trainable=True,
            scope="fc_layer_2", reuse=reuse,
            weights_initializer=tf.contrib.layers.xavier_initializer())
        prob = tf.nn.sigmoid(logit)
    return logit, prob


def generator(Z):
    with tf.variable_scope("generator") as scope:
        fc_layer_1 = tf.contrib.layers.fully_connected(
            inputs=Z, num_outputs=128, activation_fn=tf.nn.relu, trainable=True, scope="fc_layer_1",
            weights_initializer=tf.contrib.layers.xavier_initializer())
        images = tf.contrib.layers.fully_connected(
            inputs=fc_layer_1, num_outputs=784, activation_fn=tf.nn.sigmoid, trainable=True, scope="fc_layer_2",
            weights_initializer=tf.contrib.layers.xavier_initializer())
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


G_sample = generator(Z)
D_logit_real, D_real = discriminator(X)
D_logit_fake, D_fake = discriminator(G_sample, reuse=True)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
D_solver = tf.train.AdamOptimizer().minimize(D_loss,
    var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))
G_solver = tf.train.AdamOptimizer().minimize(G_loss,
    var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
if not os.path.exists('mnist_gan_conv_out/'):
    os.makedirs('mnist_gan_conv_out/')


i = 0
for it in range(200001):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig('mnist_gan_conv_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    X_mb, _ = mnist.train.next_batch(mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    if it % 1000 == 0:
        D_r, D_f = sess.run([D_real, D_fake], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        print()
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('D_accuracy: {:.4}'.format(np.concatenate((D_r, 1. - D_f)).mean()))
