from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange
from scipy.misc import imresize
from subpixel import PS

from ops import *
from utils import *

def doresize(x, shape):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, shape)
    return y

class DCGAN(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=64, image_shape=[128, 128, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 32
        self.sample_size = batch_size
        self.image_shape = image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                    name='real_images')
        try:
            self.up_inputs = tf.image.resize_images(self.inputs, self.image_shape[0], self.image_shape[1], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError:
            # newer versions of tensorflow
            self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')

        self.G = self.generator(self.inputs)

        self.G_sum = tf.image_summary("G", self.G)

        self.g_loss = tf.reduce_mean(tf.square(self.images-self.G))

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        # first setup validation data
        data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.jpg")))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.merge_summary([self.G_sum, self.g_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size,]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [8, 8], './samples/inputs_small.png')
        save_images(sample_images, [8, 8], './samples/reference.png')

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in xrange(config.epoch):
            data = sorted(glob(os.path.join("./data", config.dataset, "train", "*.jpg")))
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                input_batch = [doresize(xx, [self.input_size,]*2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                # Update G network
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                    feed_dict={ self.inputs: batch_inputs, self.images: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errG))

                if np.mod(counter, 100) == 1:
                    samples, g_loss, up_inputs = self.sess.run(
                        [self.G, self.g_loss, self.up_inputs],
                        feed_dict={self.inputs: sample_input_images, self.images: sample_images}
                    )
                    if not have_saved_inputs:
                        save_images(up_inputs, [8, 8], './samples/inputs.png')
                        have_saved_inputs = True
                    save_images(samples, [8, 8],
                                './samples/valid_%s_%s.png' % (epoch, idx))
                    print("[Sample] g_loss: %.8f" % (g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def generator(self, z):
        # project `z` and reshape
        self.h0, self.h0_w, self.h0_b = deconv2d(z, [self.batch_size, 32, 32, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0', with_w=True)
        h0 = lrelu(self.h0)

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, 32, 32, self.gf_dim], name='g_h1', d_h=1, d_w=1, with_w=True)
        h1 = lrelu(self.h1)

        h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, 32, 32, 3*16], d_h=1, d_w=1, name='g_h2', with_w=True)
        h2 = PS(h2, 4, color=True)

        return tf.nn.tanh(h2)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
