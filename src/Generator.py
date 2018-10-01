#coding=utf-8
import tensorflow as tf 
import numpy as np 
import  pickle
import time
from QACNN import QACNN


class Generator(QACNN):
    
    def __init__(self, sequence_length, batch_size, vocab_size,
                 embedding_size, filter_sizes, num_filters,
                 dropout_keep_prob=1.0, l2_reg_lambda=0.0, paras=None,
                 learning_rate=1e-2, embeddings=None, loss="pair",trainable=True):

        with tf.name_scope('QACNN_init') as scope:        
            QACNN.__init__(self, sequence_length, batch_size,
                           vocab_size, embedding_size,
                           filter_sizes, num_filters, dropout_keep_prob=dropout_keep_prob,
                           l2_reg_lambda=l2_reg_lambda, paras=paras, learning_rate=learning_rate,
                           embeddings=embeddings, loss=loss, trainable=trainable)

        self.model_type="Gen"
        self.reward  = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index = tf.placeholder(tf.int32, shape=[None], name='neg_index')

        with tf.name_scope('batch_score') as scope:        
            self.batch_scores =tf.nn.softmax( self.score13-self.score12) #~~~~~
        # self.all_logits =tf.nn.softmax( self.score13) #~~~~~
        with tf.name_scope('prob') as scope:        
            self.prob = tf.gather(self.batch_scores, self.neg_index)

        with tf.name_scope('gan_loss') as scope:        
            self.gan_loss =  -tf.reduce_mean(tf.log(self.prob) *self.reward) +l2_reg_lambda * self.l2_loss

        with tf.name_scope('gen_optimizer') as scope:        
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.gan_loss)
            self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # minize attention
        with tf.name_scope('gen_minimize_attention') as scope:        
            self.gan_score=self.score13-self.score12
            self.dns_score=self.score13
      



