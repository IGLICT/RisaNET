import sys
import os
import getopt
from six.moves import xrange
import h5py, time
import tensorflow as tf
import numpy as np

from utils import leaky_relu, leaky_relu2, batch_norm_wrapper
from data_loader import load_data_new, load_neighbour, load_labelMatrix, load_structMatrix
from meshvae import meshVAE

### Some hyper-parameters...
learning_rate = 0.00001
lambda1 = 1
lambda2 = 1
lambda3 = 1
lambda4 = 1
lambda5 = 1
bound = 1
bound_2 = 1
latent_zdim = 32
latent_zdim_2 = 32
key_dim = 32
mat = 'guitar'
epoch_num = 10000
epoch_num_2 = 10000
batch_size = 32
### Change Datapath for corresponding features...
restore_path = ''
matpath = './pre_processed_features/guitar/edgefeature.mat'
label_path = './pre_processed_features/guitar/labelMatrix.mat'
struct_path = './pre_processed_features/guitar/structMatrix.mat'

### Parsing...
opts, args = getopt.getopt(sys.argv[1:], "a:b:c:d:e:l:f:x:y:m:n:p:s:r:k:", \
	["KLpart", "Trippart", "Part2Global", "KLglobal", "Tripglobal", "learning_rate", "epoch_num", "BoundPart", "BoundGlobal", "HidePart", "HideGlobal", "matname", "batch_size","restore_path","ckpt_path"])
print(opts, args)
for op, value in opts:
    print(op, value)
    if op == "-a":
        lambda1 = float(value)
    elif op == "-b":
    	lambda2 = float(value)
    elif op == "-c":
    	lambda3 = float(value)
    elif op == "-d":
    	lambda4 = float(value)
    elif op == "-e":
        lambda5 = float(value)
    elif op == "-l":
        learning_rate = float(value)
    elif op == "-f":
        epoch_num = int(value)
    elif op == "-x":
    	bound = float(value)
    elif op == "-y":
        bound_2 = float(value)
    elif op == "-m":
        latent_zdim = int(value)
    elif op == "-n":
        latent_zdim_2 = int(value)    
    elif op == "-p":
        mat = value     
    elif op == "-s":
        batch_size = int(value)        
    elif op == "-r":
    	restore_path = value
    elif op == "-k":
    	ckpt_path = value
    else:
        sys.exit()

### Create directories to record training process...
matname = './' + mat + '.mat'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
timecurrent = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
logfolder = './' + timecurrent + "Joint" + "_l" + str(learning_rate)+'_a' + str(lambda1) + '_b' + str(lambda2) + '_c' + str(lambda3) + '_d' + str(lambda4) + "_f"+str(epoch_num_2)+'_x' + str(bound) + '_y' + str(bound_2) + '_m' + str(latent_zdim) + '_n' + str(latent_zdim_2)

##########################################################################################
class risaNET():

    def __init__(self, matpath):
        # Get splited dataset
        self.mask, self.part_num, self.modelnum, self.edgenum, self.logdr, self.e_nb, self.maxdegree, self.degree = load_data_new(matpath)
        # Set initialize parameters
        self.batch_size = batch_size
        self.bound = bound
        # Instantilize each VAE part and append them to partVAE list
        self.partVAE = []
        for i in range(self.part_num):
            print("# Making Part {}".format(i))
            self.partVAE.append(meshVAE(batch_size, label_path, latent_zdim, bound, lambda1, learning_rate, self.e_nb, self.edgenum, self.degree, self.maxdegree, part_no=i))

        # Get Label Matrix
        self.labelMatrix = load_labelMatrix(label_path)
        self.input_label = tf.placeholder(tf.float32, [self.batch_size, self.batch_size], name='label_batch')
        self.bound = bound

        # ----------------------------------------VAE Set for Stage 1
        # Get Weighted Structure Sets
        self.structMatrix = load_structMatrix(struct_path)
        self.input_struct = tf.placeholder(tf.float32, [None, 8*(self.part_num)], name='struct_batch')
        self.latent_set = []
        for i in range(self.part_num):
            self.latent_set.append(self.partVAE[i].z_mean)
        self.Wk = []
        self.Wq = []
        self.latent_struct = tf.transpose(self.latent_set, perm=[1, 0, 2])
        for p in range(self.part_num):
            self.Wk.append(tf.get_variable("W_key"+str(p), [latent_zdim, key_dim], tf.float32, tf.random_normal_initializer(stddev=0.02)))
            self.Wq.append(tf.get_variable("W_query"+str(p), [latent_zdim, key_dim], tf.float32, tf.random_normal_initializer(stddev=0.02)))
        self.s_r, self.atten_latent_struct = self.attention_mechanism( self.latent_struct, self.Wk, self.Wq, key_dim, latent_zdim)
        self.atten_latent_struct = tf.reshape(self.atten_latent_struct, shape=[tf.shape(self.partVAE[0].z_mean)[0], -1])        
        self.sgeo, self.sstruct = self.geo_struct_attention_mechanism(self.atten_latent_struct, self.input_struct, self.part_num * latent_zdim )
        self.weight_latent = tf.multiply(self.atten_latent_struct, self.sgeo)
        self.weight_struct = tf.multiply(self.input_struct, self.sstruct)
        self.weight_latent_struct = tf.concat([self.weight_latent, self.weight_struct], axis=1)        
        # Calculate Triplet loss of all VAE part
        self.distanceMatrix = self.get_l1_matrix(self.weight_latent_struct, name='l1_matrix')
        self.margin = self.distanceMatrix - self.bound
        self.standard = self.margin * self.input_label
        self.sigmoidresult = tf.sigmoid(self.standard)
        self.total_triplet = tf.reduce_sum(self.standard)
        
        # Calculate generation and KL loss of all VAE part
        self.generation_loss_set = []
        for i in range(self.part_num):
            self.generation_loss_set.append(self.partVAE[i].generation_loss)
        self.total_generation = tf.reduce_sum(self.generation_loss_set)
        
        self.KL_loss_set = []
        for i in range(self.part_num):
            self.KL_loss_set.append(self.partVAE[i].KL_loss)
        self.total_KL = tf.reduce_sum(self.KL_loss_set)
       
        # ------------------------------------------ VAE for stage 2
        self.hiddendim_2 = latent_zdim_2
        self.embedding_inputs = tf.placeholder(tf.float32, [None, self.hiddendim_2], name = 'embedding_inputs')
        self.encode, self.encode_std, self.encode_gauss, self.decode, self.tencode, self.tencode_std, self.tencode_gauss, self.tdecode = self.vae_struct(self.weight_latent_struct, self.embedding_inputs, self.hiddendim_2, name='vae_struct')

		# Calculate total loss and Optimization Function for stage 2
        self.generation_loss_2 = 1 * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(self.weight_latent_struct - self.decode, 2), axis=1))
        self.KL_loss_2 = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.encode) + tf.square(self.encode_std) - tf.log(1e-8 + tf.square(self.encode_std)) - 1, axis=1))
        self.bound_2 = bound_2
        self.distanceMatrix_2 = self.get_l2_matrix(self.encode, name='l1_matrix_2')
        self.margin_2 = self.distanceMatrix_2 - self.bound_2
        self.standard_2 = self.margin_2 * self.input_label
        self.sigmoidresult_2 = tf.sigmoid(self.standard_2)
        self.triplet_loss_2 = tf.reduce_sum(self.standard_2)
        
        # Calculate total loss and Optimization Function
        self.cost_stg1 = self.total_generation + lambda1 * self.total_KL + lambda2 * self.total_triplet
        self.cost_stg2 = lambda3 * self.generation_loss_2 + lambda4 * self.KL_loss_2 + lambda5 * self.triplet_loss_2
        # self.optimizer_stg1 = tf.train.AdamOptimizer(learning_rate).minimize(self.cost_stg1)     
        # self.update_stg2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vae_struct")
        # self.optimizer_stg2 = tf.train.AdamOptimizer(learning_rate).minimize(self.cost_stg2, var_list=self.update_stg2)
        self.cost = self.cost_stg1 + self.cost_stg2
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)   

        tf.summary.histogram("DisMat", self.distanceMatrix)
        tf.summary.scalar('total_generation', self.total_generation)
        tf.summary.scalar("total_KL", self.total_KL)
        tf.summary.scalar("total_triplet", self.total_triplet)
        tf.summary.scalar('cost', self.cost_stg1)
        
        tf.summary.histogram("DisMat2", self.distanceMatrix_2)
        tf.summary.scalar('generation2', self.generation_loss_2)
        tf.summary.scalar("KL2", self.KL_loss_2)
        tf.summary.scalar("triplet2", self.triplet_loss_2)
        tf.summary.scalar('cost2', self.cost_stg2)
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=2)

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

    def attention_mechanism(self, feature, Wk, Wq, key_dim, latent_zdim):
        with tf.variable_scope("attention_mechanism", reuse=tf.AUTO_REUSE) as scope:
            batch_size = tf.shape(feature)[0] 
            Key =[] # part* [batch, key]
            Query = tf.zeros([batch_size, key_dim]) #[batch, key]
            for p in range(self.part_num):
                wk = Wk[p] #[latent, key]
                wq = Wq[p] #[latent, key]
                f = feature[:, p, :]
                f = tf.squeeze(f) #[batch, latent]
                Key.append(tf.matmul(f, wk)) #[batch, key]
                Query = Query + tf.matmul(f, wk) #[batch, key]
            Query = tf.expand_dims(Query, axis=2) #[batch, key, 1]
            Key = tf.reshape(Key, [self.part_num, batch_size, key_dim]) #[part, batch, key]
            Key = tf.transpose(Key, perm=[1, 0, 2]) #[batch, part, key]
            Value = tf.squeeze(tf.matmul(Key, Query)) #[batch, part]
            Score = tf.exp(Value) / (1e-10 + tf.reduce_sum(tf.exp(Value), axis=1, keepdims=True)) #[batch, part]
            Score_t = tf.tile(tf.expand_dims(Score, axis=2), [1, 1, latent_zdim]) #[batch, part, latent]
            Result = tf.multiply(feature, Score_t)
            return Score, Result

    def geo_struct_attention_mechanism(self, geo_feature, struct_feature, input_dim):
        with tf.variable_scope("geo_struct_attention_mechanism", reuse=tf.AUTO_REUSE) as scope:
            Wgeo1 = tf.get_variable("Wgeo1", [input_dim, latent_zdim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            Wstruct1 = tf.get_variable("Wstruct1", [(self.part_num) * 8, latent_zdim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            Wgeo2 = tf.get_variable("Wgeo2", [latent_zdim, 1], tf.float32, tf.random_normal_initializer(stddev=0.02))
            Wstruct2 = tf.get_variable("Wstruct2", [latent_zdim, 1], tf.float32, tf.random_normal_initializer(stddev=0.02))
            batch_size = tf.shape(geo_feature)[0]

            Vgeo1 = tf.matmul(geo_feature, Wgeo1)
            Vstruct1 = tf.matmul(struct_feature, Wstruct1)
            Vgeo2 = tf.matmul(Vgeo1, Wgeo2)
            Vstruct2 = tf.matmul(Vstruct1, Wstruct2)

            Value = tf.concat([Vgeo2, Vstruct2], axis=1)
            Score = tf.exp(Value) / (1e-10 + tf.reduce_sum(tf.exp(Value), axis=1, keepdims=True))
            Sgeo = Score[:, 0]
            Sgeo = tf.expand_dims(Sgeo, 1)
            Sgeo_t = tf.tile(Sgeo, [1, input_dim])
            Sstruct = Score[: ,1]
            Sstruct = tf.expand_dims(Sstruct, 1)
            Sstruct_t = tf.tile(Sgeo, [1, (self.part_num) * 8])
            tf.summary.histogram("Geo_Score", Sgeo)
            tf.summary.histogram("Struct_Score", Sstruct)

            return Sgeo_t, Sstruct_t

    def encoder_symm(self, input_mesh, training = True, keep_prob = 1.0):
        with tf.variable_scope("encoder_symm") as scope:
            if(training == False):
                keep_prob = 1.0
                scope.reuse_variables()

            bn = True

            matrix1, bias1, h1 = self.linear(input_mesh, self.part_num*latent_zdim+(self.part_num)*8, 256, name = 'fc_1', training = training, special_activation = False, bn = bn)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)

            matrix2, bias2, h2 = self.linear(h1, 256, 128, name = 'fc_2', training = training, special_activation = False, bn = bn)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)

            matrix3, bias3, h3 = self.linear(h2, 128, 64, name = 'fc_3', training = training, special_activation = False, bn = bn)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)
            _, _, mean = self.linear(h3, 64, self.hiddendim_2, name = 'mean', training = training, no_activation = True, bn = False)
            _, _, stddev = self.linear(h3, 64, self.hiddendim_2, name = 'stddev', training = training, no_activation = True, bn = False)
            stddev = tf.sqrt(tf.nn.softsign(stddev)+1.0)

        return mean, stddev

    def decoder_symm(self, z, training = True, keep_prob = 1.0):
        with tf.variable_scope("decoder_symm") as scope:
            if(training == False):
                keep_prob = 1.0
                scope.reuse_variables()

            bn = True

            matrix1, bias1, h1 = self.linear(z, self.hiddendim_2, 64, name = 'fc_1', training = training, special_activation = False, bn = bn)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)

            matrix2, bias2, h2 = self.linear(h1, 64, 128, name = 'fc_2', training = training, special_activation = False, bn = bn)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)

            matrix3, bias3, h3 = self.linear(h2, 128, 256, name = 'fc_3', training = training, special_activation = False, bn = bn)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)

            matrix3, bias3, output = self.linear(h3, 256, self.part_num*latent_zdim+(self.part_num)*8, name = 'fc_4', training = training, no_activation = True, bn = False)

        return output
    def leaky_relu(self, input_, alpha = 0.1):
        return tf.nn.leaky_relu(input_)

    def linear(self, input_, input_size, output_size, name='Linear', training = True, special_activation = False, no_activation = False, bn = True, stddev=0.02, bias_start=0.0):
        with tf.variable_scope(name) as scope:

            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            matrix = tf.get_variable("weights", [input_size, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size], tf.float32, initializer=tf.constant_initializer(bias_start))
            output = tf.matmul(input_, matrix) + bias

            if bn == False:
                fb = output
            else:
                fb = batch_norm_wrapper(output, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)

        return matrix, bias, fa

    def vae_struct(self, input, embedding_inputs, hiddendim_2, name='vae_struct'):
        with tf.variable_scope(name) as scope:
            encode, encode_std = self.encoder_symm(input, training = True)
            encode_gauss = encode + encode_std * embedding_inputs
            decode = self.decoder_symm(encode_gauss, training = True)
            tencode, tencode_std = self.encoder_symm(input, training = False)
            tencode_gauss = tencode + tencode_std * embedding_inputs
            tdecode = self.decoder_symm(tencode_gauss, training = False)
            return encode, encode_std, encode_gauss, decode, tencode, tencode_std, tencode_gauss, tdecode

    def train(self, restore=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not os.path.isdir(logfolder):
            os.mkdir(logfolder)

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            if not restore is None:
                self.saver.restore(sess, restore)
            file = open(logfolder + '/' + '_script_result.txt', 'w')
            if not os.path.isdir(logfolder+ '/tb'):
                os.mkdir(logfolder+ '/tb')
            for epoch in xrange(0, epoch_num):
                rand_index = np.random.choice(list(range(0,len(self.logdr[0]),5))+list(range(1,len(self.logdr[0]),5))+list(range(2,len(self.logdr[0]),5))+list(range(3,len(self.logdr[0]),5)), size=self.batch_size)
                input_label = np.zeros(shape=(len(rand_index), len(rand_index)))
                for i in range(len(rand_index)):
                    row = rand_index[i]
                    row = row.repeat(len(rand_index))
                    col = rand_index
                    input_label[i] = self.labelMatrix[row, col]

                timecurrent1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                eps_s = np.random.normal(size=(len(rand_index), latent_zdim))
                eps_logdr = np.random.normal(size=(len(rand_index), latent_zdim))
                teps_s = np.zeros(shape=(len(rand_index), latent_zdim))
                teps_logdr = np.zeros(shape=(len(rand_index), latent_zdim))                
                eps_input = np.random.normal(size=(len(rand_index), latent_zdim_2))
                feed_dict_ofall = {self.input_label: input_label, self.input_struct: self.structMatrix[rand_index], self.embedding_inputs: eps_input}

                for i in range(self.part_num):
                    feed_input_mask = self.mask[i][rand_index]
                    feed_input_mask = np.expand_dims(feed_input_mask, 1)
                    feed_dict_ofall[self.partVAE[i].inputs_logdr] = self.logdr[i][rand_index]
                    feed_dict_ofall[self.partVAE[i].input_mask] = feed_input_mask
                    feed_dict_ofall[self.partVAE[i].eps_logdr] = eps_logdr
                    feed_dict_ofall[self.partVAE[i].eps_logdr_test] = teps_logdr                    
                    feed_dict_ofall[self.partVAE[i].input_label] = input_label
              
                _,gen, KL, trip,cost1,gen2, KL2, trip2, cost2, cost_al = sess.run([self.optimizer, self.total_generation, self.total_KL, self.total_triplet, self.cost_stg1, self.generation_loss_2, self.KL_loss_2, self.triplet_loss_2, self.cost_stg2, self.cost],
                                           feed_dict=feed_dict_ofall)
  
                # ================Save checkpoint, write logger, write summary
                # if np.mod(epoch + 1, 200) == 0 and epoch != 0:
                #     self.saver.save(sess, logfolder + '/' + 'meshvae.model', global_step=epoch + 1)
                #     # self.save_z(logfolder + '/meshvae.model-' + str(epoch+1), logfolder, epoch+1)

                if np.mod(epoch + 1, 50) == 0:
                    print("%s Epoch: [%4d]G: %.4f K: %.4f T: %.4f cost1: %.4f G2: %.4f K2: %.4f T2: %.4f cost2: %.4f cost: %.8f\n"
                          % (timecurrent1, epoch + 1, gen, KL, trip, cost1, gen2, KL2, trip2, cost2, cost_al))
                file.write("%s Epoch: [%4d]G: %.4f K: %.4f T: %.4f cost1: %.4f G2: %.4f K2: %.4f T2: %.4f cost2: %.4f cost: %.8f\n"
                      % (timecurrent1, epoch + 1, gen, KL, trip, cost1, gen2, KL2, trip2, cost2, cost_al))

        return

    def save_z(self, restore, foldername, times=0):
        print('###Loading...')
        with tf.Session() as sess:
            self.saver.restore(sess, restore)
            index = list(xrange(len(self.logdr[0])))
            eps_s = np.zeros(shape=(len(index), latent_zdim))
            eps_logdr = np.zeros(shape=(len(index), latent_zdim))
            teps_s = np.zeros(shape=(len(index), latent_zdim))
            teps_logdr = np.zeros(shape=(len(index), latent_zdim))
            eps_input = np.zeros(shape=(len(index), latent_zdim_2))
            teps_input = np.zeros(shape=(len(index), latent_zdim_2))               
            feed_dict_ofall = {self.input_struct: self.structMatrix[index], self.embedding_inputs: eps_input}
            for i in range(self.part_num):
                feed_input_mask = self.mask[i][index]
                feed_input_mask = np.expand_dims(feed_input_mask, 1)
                feed_dict_ofall[self.partVAE[i].inputs_logdr] = self.logdr[i][index]
                feed_dict_ofall[self.partVAE[i].input_mask] = feed_input_mask
                feed_dict_ofall[self.partVAE[i].eps_logdr] = eps_logdr
                feed_dict_ofall[self.partVAE[i].eps_logdr_test] = teps_logdr                    

            z = sess.run([self.encode],
                        feed_dict=feed_dict_ofall)
            z = np.squeeze(z)

            print('###Writing...')
            name = foldername + '/'  + str(times) +'test_index.h5' 
            print(name)
            f = h5py.File(name, 'w')
            f['feature_vector'] = z
            f.close()              
        return

##########################################################################################
def main():
    risanet = risaNET(matpath)
    if restore_path:
        risanet.save_z(restore_path+'/meshvae.model-'+ckpt_path, restore_path)    	
    else:
        risanet.train()
        risanet.save_z(logfolder + '/meshvae.model-' + str(epoch_num), logfolder)


if __name__ == '__main__':
    main()
