from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

class Autoencoder:
    def __init__(self, D, d = 32):
        self.D = D
        self.d = d
        self.autoencoder_lr = FLAGS.autoencoder_lr
        self.loss_func = mse
        self.train_phase = tf.placeholder(dtype=tf.bool)

        self.dim_hidden = [256, 128, 64, self.d, 64, 128, 256]
        self.forward= self.forward_fc
        self.construct_weights = self.construct_fc_weights

    def construct_autoencoder(self, input_tensors=None, prefix = 'autoencoder_train'):
        if input_tensors is None:
            self.X = tf.placeholder(tf.float32)
            self.Y = tf.placeholder(tf.float32)

        else:
            self.X = input_tensors['X']
            self.Y = input_tensors['Y']

        with tf.variable_scope('autoencoder', reuse=None) as training_scope:

            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            def task_learn(inp, reuse=True):
                # L_Ti
                task_output = self.forward(inp, weights, reuse=reuse)
                task_loss = self.loss_func(task_output, inp)

                grads = tf.gradients(task_loss, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.autoencoder_lr * gradients[key] for key in weights.keys()]))
                output = [task_output, task_loss]

                return output

            def task_val(inp, reuse=True):
                # L_Ti
                task_output = self.forward(inp, weights, reuse=reuse)
                task_loss = self.loss_func(task_output, inp)
                output = [task_output, task_loss]
                return output


            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_learn((self.X[0]), False)

            out_dtype = [tf.float32]
            result = task_learn((self.X), True)
            result_val = task_val((self.Y), True)
            output, loss = result
            output_val, loss_val = result_val

        ## Performance & Optimization
        if 'train' in prefix:
            self.output = output
            self.loss = loss

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'autoencoder')):
                self.train_op = tf.train.AdamOptimizer(self.autoencoder_lr).minimize(loss, var_list=tf.trainable_variables())

        self.output_val = output_val
        self.loss_val = loss_val


        ## Summaries
        # tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        # if self.classification:
        #     tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        # for j in range(num_updates):
        # tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            # if self.classification:
            #     tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.D, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.D], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.D]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.tensordot(inp, weights['w1'],1) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0', is_training = self.train_phase)
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.tensordot(hidden, weights['w'+str(i+1)], 1) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1), is_training = self.train_phase)
        return tf.tensordot(hidden, weights['w'+str(len(self.dim_hidden)+1)], 1) + weights['b'+str(len(self.dim_hidden)+1)]

    def encode(self, inp, reuse=True):
        with tf.variable_scope('autoencoder', reuse=reuse) as training_scope:
            inp = tf.reshape(inp, [1, -1])
            hidden = normalize(tf.matmul(inp, self.weights['w1']) + self.weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0', is_training = self.train_phase)
            for i in range(1,int(len(self.dim_hidden)/2)):
                hidden = normalize(tf.matmul(hidden, self.weights['w'+str(i+1)]) + self.weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1), is_training = self.train_phase)
            x = tf.matmul(hidden, self.weights['w'+str(int(len(self.dim_hidden)/2)+1)]) + self.weights['b'+str(int(len(self.dim_hidden)/2)+1)]
        return x

    # def encode(self, inp, reuse=False):
    #     hidden = tf.tensordot(inp, self.weights['w1'], 1) + self.weights['b1']
    #     for i in range(1,int(len(self.dim_hidden)/2)):
    #         hidden = tf.tensordot(hidden, self.weights['w'+str(i+1)], 1) + self.weights['b'+str(i+1)]
    #     return tf.tensordot(hidden, self.weights['w'+str(int(len(self.dim_hidden)/2)+1)], 1) + self.weights['b'+str(int(len(self.dim_hidden)/2)+1)]
    #

