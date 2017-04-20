import tensorflow.contrib.layers as layers
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
import os
import requests
import gzip
import cPickle as pickle
import numpy as np

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def G(z, batch_size):
   z = layers.fully_connected(z, 4*4*1024, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])

   conv1 = layers.convolution2d_transpose(z, 512, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu, scope='g_conv1')

   conv2 = layers.convolution2d_transpose(conv1, 256, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu, scope='g_conv2')
   
   conv3 = layers.convolution2d_transpose(conv2, 128, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu, scope='g_conv3')

   conv4 = layers.convolution2d_transpose(conv3, 1, 5, stride=2, activation_fn=tf.nn.tanh, scope='g_conv4')
   
   conv4 = conv4[:,:28,:28,:]
   return conv4

def D(x, reuse=False):
   
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = layers.conv2d(x, 64, 5, stride=2, activation_fn=None, scope='d_conv1')
      conv1 = lrelu(conv1)

      conv2 = layers.conv2d(conv1, 128, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv2')
      conv2 = lrelu(conv2)

      conv3 = layers.conv2d(conv2, 256, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv3')
      conv3 = lrelu(conv3)

      conv4 = layers.conv2d(conv3, 512, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv4')
      conv4 = lrelu(conv4)
      
      conv5 = layers.conv2d(conv4, 1, 4, stride=1, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv5')
      conv5 = lrelu(conv5)

      fc1 = layers.fully_connected(layers.flatten(conv5), 1, scope='d_fc1', activation_fn=None)
      fc1 = tf.nn.sigmoid(fc1)

   return fc1


def train(mnist_train):
   with tf.Graph().as_default():
     
      batch_size = 128

      # placeholder to keep track of the global step
      global_step = tf.Variable(0, trainable=False, name='global_step')
      
      # placeholder for mnist images
      images = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='images')
      
      # placeholder for the latent z vector
      z = tf.placeholder(tf.float32, [batch_size, 100], name='z')

      # generate an image from noise prior z
      generated_images = G(z, batch_size)

      # small weight factor so D doesn't go to 0
      e = 1e-12

      # loss of D on real images
      D_real = D(images)
      D_fake = D(generated_images)

      # final objective function for D
      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))

      # instead of minimizing (1-D(G(z)), maximize D(G(z))
      errG = tf.reduce_mean(-tf.log(D_fake + e))
   
      # get all trainable variables, and split by network G and network D
      t_vars = tf.trainable_variables()
      d_vars = [var for var in t_vars if 'd_' in var.name]
      g_vars = [var for var in t_vars if 'g_' in var.name]

      # training operators for G and D
      G_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step)
      D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(errD, var_list=d_vars)

      saver = tf.train.Saver(max_to_keep=1)
   
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
      sess.run(init)
      
      # tensorboard summaries
      try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
      except:pass
      try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
      except:pass

      # write out logs for tensorboard to the checkpointSdir
      summary_writer = tf.summary.FileWriter('checkpoints/dcgan/logs/', graph=tf.get_default_graph())

      # load previous checkpoint if there is one
      ckpt = tf.train.get_checkpoint_state('checkpoints/dcgan/')
      if ckpt and ckpt.model_checkpoint_path:
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model restored'
         except:
            print 'Could not restore model'
            pass


      merged_summary_op = tf.summary.merge_all()
      # training loop
      step = sess.run(global_step)
      num_train = len(mnist_train)
      while True:
         s = time.time()
         step += 1

         epoch_num = step/(num_train/batch_size)

         # get random images from the training set
         batch_images = random.sample(mnist_train, batch_size)
         
         # generate z from a normal distribution between [-1, 1] of length 100
         batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)

         # run D
         sess.run(D_train_op, feed_dict={z:batch_z, images:batch_images})
         
         # run G
         sess.run(G_train_op, feed_dict={z:batch_z, images:batch_images})
         sess.run(G_train_op, feed_dict={z:batch_z, images:batch_images})

         # get losses WITHOUT running the networks
         G_loss, D_loss, summary = sess.run([errG, errD, merged_summary_op], feed_dict={z:batch_z, images:batch_images})
         summary_writer.add_summary(summary, step)
         
         if step%10==0:print 'epoch:',epoch_num,'step:',step,'G loss:',G_loss,' D loss:',D_loss,' time:',time.time()-s

         if step%500 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, 'checkpoints/dcgan/checkpoint-', global_step=global_step)

            # generate some to write out
            batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
            gen_imgs = np.asarray(sess.run(generated_images, feed_dict={z:batch_z, images:batch_images}))
            random.shuffle(gen_imgs)
            # write out a few (10)
            c = 0
            for img in gen_imgs:
               img = np.reshape(img, [28, 28])
               plt.imsave('checkpoints/dcgan/images/0000'+str(step)+'_'+str(c)+'.png', img)
               if c == 5:
                  break
               c+=1


if __name__ == '__main__':

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir('checkpoints/dcgan/')
   except: pass
   try: os.mkdir('checkpoints/dcgan/images/')
   except: pass
   try: os.mkdir('checkpoints/dcgan/logs/')
   except: pass
   
   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

   # check if it's already downloaded
   if not os.path.isfile('mnist.pkl.gz'):
      print 'Downloading mnist...'
      with open('mnist.pkl.gz', 'wb') as f:
         r = requests.get(url)
         if r.status_code == 200:
            f.write(r.content)
         else:
            print 'Could not connect to ', url

   print 'opening mnist'
   f = gzip.open('mnist.pkl.gz', 'rb')
   train_set, val_set, test_set = pickle.load(f)

   mnist_train = []

   # we will be using all splits. This goes through and adds a dimension
   # to the images and adds them to a training set
   for t,l in zip(*train_set):
      mnist_train.append(np.reshape(t, (28, 28, 1)))
   for t,l in zip(*val_set):
      mnist_train.append(np.reshape(t, (28, 28, 1)))
   for t,l in zip(*test_set):
      mnist_train.append(np.reshape(t, (28, 28, 1)))

   mnist_train = np.asarray(mnist_train)

   train(mnist_train)
