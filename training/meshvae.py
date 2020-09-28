from six.moves import xrange
import h5py, time
import tensorflow as tf
import numpy as np

from utils import leaky_relu, leaky_relu2, batch_norm_wrapper
from data_loader import load_data_new, load_neighbour, load_labelMatrix, load_structMatrix
# from risanet import label_path

class meshVAE():
    def __init__(self, batch_size, label_path, latent_zdim, bound, lambda1, learning_rate, e_neighbour, edgenum, degree, maxdegree, part_no, padding=False, result_max1=0.9, result_min1=-0.9):
        self.part_no = part_no
        self.batch_size = batch_size
        self.e_neighbour = e_neighbour 
        self.degree = degree 
        self.edgenum = edgenum 
        self.maxdegree = maxdegree
        self.labelMatrix = load_labelMatrix(label_path)

        self.edges = self.edgenum

        if padding == True:
            self.edges += 1
            self.is_padding = True
        else:
            self.is_padding = False

        self.inputs_logdr = tf.placeholder(tf.float32, [None, self.edgenum, 2], name='input_logdr'+str(self.part_no))
        self.input_mask = tf.placeholder(tf.float32, [None, 1], name='input_mask'+str(self.part_no))
        self.g_z_logdr = tf.placeholder(tf.float32, [None, latent_zdim], name='g_z_logdr'+str(self.part_no))
        self.input_label = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name='label_batch')
        self.bound = bound

        self.e_nb = tf.constant(self.e_neighbour, dtype='int32', shape=[self.edges, 4], name='e_nb_relation'+str(self.part_no))
        self.degrees = tf.constant(self.degree, dtype='float32', shape=[self.edgenum, 1], name='degrees'+str(self.part_no))

        self.logdr0_n1, self.logdr_n1, self.logdr_e1 = self.get_conv_weights(2, 2, name='logdr_convw1'+str(self.part_no))
        self.logdr0_n2, self.logdr_n2, self.logdr_e2 = self.get_conv_weights(2, 2, name='logdr_convw2'+str(self.part_no))
        self.logdr0_n3, self.logdr_n3, self.logdr_e3 = self.get_conv_weights(2, 2, name='logdr_convw3'+str(self.part_no))

        ##-------------- Latent vector of logdr and s --------------##
        self.hidden_dim = latent_zdim
        self.fcparams_logdr = tf.get_variable("fcparams_logdr"+str(self.part_no), [self.edges * 2, self.hidden_dim], tf.float32,
                                              tf.random_normal_initializer(stddev=0.02))
        self.stdparams_logdr = tf.get_variable("stdparams_logdr"+str(self.part_no), [self.edges * 2, self.hidden_dim], tf.float32,
                                               tf.random_normal_initializer(stddev=0.02))
        self.z_mean_logdr, self.z_std_logdr = self.encoder_logdr(self.inputs_logdr, self.input_mask, training=True)
        self.z_mean = self.z_mean_logdr

        self.eps_logdr = tf.placeholder(tf.float32, [None, latent_zdim], name='eps_logdr'+str(self.part_no))
        self.decoder_input_logdr = self.z_mean_logdr + self.z_std_logdr * self.eps_logdr

        self.generated_mesh_train_logdr = self.decoder_logdr(self.decoder_input_logdr, self.input_mask, training=True)

        self.z_mean_logdr_test, self.z_std_logdr_test = self.encoder_logdr(self.inputs_logdr, self.input_mask, training=False)
        self.eps_logdr_test = tf.placeholder(tf.float32, [None, latent_zdim], name='eps_logdr_test'+str(self.part_no))
        self.decoder_input_logdr_test = self.z_mean_logdr_test + self.z_std_logdr_test * self.eps_logdr_test

        self.tz_mean = self.z_mean_logdr_test
        self.test_mesh_logdr = self.decoder_logdr(self.decoder_input_logdr_test, self.input_mask, training=False)

        self.g_mesh_logdr = self.decoder_logdr(self.g_z_logdr, self.input_mask, training=False)

        ##---------------- Loss function of VAE--------------##
        self.generation_loss = 1 * 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.pow(self.inputs_logdr - self.generated_mesh_train_logdr, 2), [1, 2]))
        self.KL_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.z_mean_logdr) + tf.square(self.z_std_logdr) - tf.log(
                1e-8 + tf.square(self.z_std_logdr)) - 1, 1)) 
        self.tgeneration_loss = 1 * 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.pow(self.inputs_logdr - self.test_mesh_logdr, 2), [1, 2])) 
        self.tKL_loss = 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(self.z_mean_logdr_test) + tf.square(self.z_std_logdr_test) - tf.log(
                1e-8 + tf.square(self.z_std_logdr_test)) - 1, 1)) 
        self.distanceMatrix = self.get_l2_matrix(self.z_mean, name='l1_matrix'+str(self.part_no))
        self.margin = self.distanceMatrix - self.bound
        self.standard = (self.margin) * self.input_label
        self.sigmoidresult = tf.sigmoid(self.standard)
        self.triplet_loss = tf.reduce_sum(self.standard)
        
        tf.summary.scalar('pgene'+str(part_no), self.generation_loss)
        tf.summary.scalar("pKL"+str(part_no), self.KL_loss)
        tf.summary.scalar("ptrip"+str(part_no), self.triplet_loss)
        
        self.total_loss = self.generation_loss + lambda1 * self.KL_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)

    def get_l1_matrix(self, feature, name='l1_matrix'):
        with tf.variable_scope(name) as scope:
            a = tf.tile(feature[0], [self.batch_size])
            a = tf.reshape(a, [self.batch_size, -1])
            for i in range(1, self.batch_size):
                tmp = tf.tile(feature[i], [self.batch_size])
                tmp = tf.reshape(tmp, [self.batch_size, -1])
                a = tf.concat([a, tmp], 0)

            b = tf.tile(feature, [self.batch_size, 1])

            abs_val = tf.abs(a - b)

            sum_abs = tf.reduce_sum(abs_val, 1)

            l1_matrix = tf.reshape(sum_abs, [self.batch_size, self.batch_size])
            l1_matrix = tf.nn.l2_normalize(l1_matrix, dim=-1)
            return l1_matrix

    def get_l2_matrix(self, feature, name='l2_matrix'):
        with tf.variable_scope(name) as scope:
            r = tf.reduce_sum(feature * feature, 1)
            r = tf.reshape(r, [-1, 1])
            distanceMatrix = r - 2 * tf.matmul(feature, tf.transpose(feature)) + tf.transpose(r)
            distanceMatrix = tf.nn.l2_normalize(distanceMatrix, dim=-1)
            return distanceMatrix

    def get_conv_weights(self, input_dim, output_dim, name='convweight'):
        with tf.variable_scope(name) as scope:
            n0 = tf.get_variable("nb0_weights"+str(self.part_no), [input_dim, output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=0.02))
            n = tf.get_variable("nb_weights"+str(self.part_no), [input_dim, output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=0.02))            
            e = tf.get_variable("edge_weights"+str(self.part_no), [input_dim, output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=0.02))
            return n0, n, e

    def linear(self, input_, input_size, output_size, name='Linear', stddev=0.02, bias_start=0.0):
        with tf.variable_scope(name) as scope:
            matrix = tf.get_variable("weights"+str(self.part_no), [input_size, output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias"+str(self.part_no), [output_size], tf.float32,
                                   initializer=tf.constant_initializer(bias_start))

            return tf.matmul(input_, matrix) + bias  # , matrix

    def newconvlayer(self, input_feature, nb, input_dim, output_dim, nb0_weights, nb1_weights, edge_weights, name='meshconv',
                     on_edge=True, degrees=4.0, training=True, special_activation=False, no_activation=False, bn=True,
                     padding=False):
        with tf.variable_scope(name) as scope:
            if on_edge == False:
                padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)
                padded_input = tf.concat([padding_feature, input_feature], 1)
            else:
                padded_input = input_feature
            nb0 = nb[:, :2]
            nb1 = nb[:, 2:]

            def compute_nb0_feature(input_feature):
                return tf.gather(input_feature, nb0)

            def compute_nb1_feature(input_feature):
                return tf.gather(input_feature, nb1)

            total_nb0_feature = tf.map_fn(compute_nb0_feature, padded_input)
            total_nb0_feature = 2 * tf.reduce_sum(total_nb0_feature, axis=2) / degrees

            total_nb1_feature = tf.map_fn(compute_nb1_feature, padded_input)
            total_nb1_feature = 2 * tf.reduce_sum(total_nb1_feature, axis=2) / degrees

            nb0_bias = tf.get_variable("nb0_bias"+str(self.part_no), [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            nb0_feature = tf.tensordot(total_nb0_feature, nb0_weights, [[2], [0]]) + nb0_bias

            nb1_bias = tf.get_variable("nb1_bias"+str(self.part_no), [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            nb1_feature = tf.tensordot(total_nb1_feature, nb1_weights, [[2], [0]]) + nb1_bias

            edge_bias = tf.get_variable("edge_bias"+str(self.part_no), [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2], [0]]) + edge_bias

            total_feature = edge_feature + nb0_feature + nb1_feature

            if bn == False:
                fb = total_feature
            else:
                fb = batch_norm_wrapper(total_feature, is_training=training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = leaky_relu(fb)
            else:
                fa = leaky_relu2(fb)

            if padding == True:
                padding_feature = tf.zeros([tf.shape(fa)[0], 1, output_dim], tf.float32)

                _, true_feature = tf.split(fa, [1, self.edges - 1], 1)

                fa = tf.concat([padding_feature, true_feature], 1)

            return fa

    def encoder_logdr(self, input_feature, mask, training=True, padding=False):
        with tf.variable_scope("encoder_logdr"+str(self.part_no)) as scope:
            if (training == False):
                scope.reuse_variables()
            all_mask = tf.tile(mask, [1, self.hidden_dim])
            conv1 = self.newconvlayer(input_feature, self.e_nb, 2, 2, self.logdr0_n1, self.logdr_n1, self.logdr_e1, name='logdr_conv1'+str(self.part_no),
                                      training=training, padding=padding)
            conv2 = self.newconvlayer(conv1, self.e_nb, 2, 2, self.logdr0_n2, self.logdr_n2, self.logdr_e2, name='logdr_conv2'+str(self.part_no),
                              training=training, padding=padding)
            conv3 = self.newconvlayer(conv2, self.e_nb, 2, 2, self.logdr0_n3, self.logdr_n3, self.logdr_e3, name='logdr_conv3'+str(self.part_no),
                                      training=training, padding=padding, special_activation=False, bn=False)
            x0 = tf.reshape(conv3, [tf.shape(conv3)[0], self.edges * 2])
            mean = tf.matmul(x0, self.fcparams_logdr)
            std = 2 * tf.sigmoid(tf.matmul(x0, self.stdparams_logdr))
            mean = tf.multiply(mean, all_mask)
            std = tf.multiply(std, all_mask)
        return mean, std

    def decoder_logdr(self, z, mask, training=True, padding=False):
        with tf.variable_scope("decoder_logdr"+str(self.part_no)) as scope:
            if (training == False):
                scope.reuse_variables()
            h1 = tf.matmul(z, tf.transpose(self.fcparams_logdr))
            x0 = tf.reshape(h1, [tf.shape(h1)[0], self.edges, 2])

            conv1 = self.newconvlayer(x0, self.e_nb, 2, 2, tf.transpose(self.logdr0_n3), tf.transpose(self.logdr_n3), tf.transpose(self.logdr_e3),
                                      name='logdr_conv1'+str(self.part_no), training=training, padding=padding)
            conv2 = self.newconvlayer(conv1, self.e_nb, 2, 2, tf.transpose(self.logdr0_n2), tf.transpose(self.logdr_n2), tf.transpose(self.logdr_e2),
                                      name='logdr_conv2'+str(self.part_no), training=training, padding=padding)
            conv3 = self.newconvlayer(conv2, self.e_nb, 2, 2, tf.transpose(self.logdr0_n1), tf.transpose(self.logdr_n1), tf.transpose(self.logdr_e1),
                                      name='logdr_conv3'+str(self.part_no), training=training, padding=padding, special_activation=False,
                                      bn=False)

            output = conv3

            output = tf.nn.tanh(output)
            all_mask = tf.tile(mask, [1, self.edgenum])
            all_mask = tf.expand_dims(all_mask, axis=2)
            all_mask = tf.tile(all_mask, [1, 1, 2])
            output = tf.multiply(output, all_mask)
        return output