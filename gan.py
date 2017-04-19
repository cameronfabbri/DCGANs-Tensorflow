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

def lrelu(x, leak=0.2):
   return tf.maximum(leak*x, x)

def G(z):
   #g_fc1 = layers.fully_connected(z, 256, activation_fn=tf.nn.relu, scope='g_fc1')
   #g_fc2 = layers.fully_connected(g_fc1, 512, activation_fn=tf.nn.relu, scope='g_fc2')
   #g_fc3 = layers.fully_connected(g_fc2, 784, activation_fn=tf.nn.tanh, scope='g_fc3')

   
   g_fc1 = layers.fully_connected(z, 1200, activation_fn=tf.nn.relu, scope='g_fc1')
   g_fc2 = layers.fully_connected(g_fc1, 1200, activation_fn=tf.nn.relu, scope='g_fc2')
   g_fc3 = layers.fully_connected(g_fc2, 784, activation_fn=tf.nn.tanh, scope='g_fc3')
   
   print 'z:',z
   print 'g_fc1:',g_fc1
   print 'g_fc2:',g_fc2
   print 'g_fc3:',g_fc3
   return g_fc3


def D(x):
   #d_fc1 = layers.fully_connected(x, 784, activation_fn=tf.nn.relu, scope='d_fc1')
   #d_fc2 = layers.fully_connected(d_fc1, 512, activation_fn=tf.nn.relu, scope='d_fc2')
   #d_fc3 = layers.fully_connected(d_fc2, 256, activation_fn=tf.nn.relu, scope='d_fc3')
   #d_fc4 = layers.fully_connected(d_fc3, 1, activation_fn=tf.nn.sigmoid, scope='d_fc4')

   d_fc1 = layers.fully_connected(x, 240, activation_fn=None, scope='d_fc1')
   d_fc1 = lrelu(d_fc1)
   
   d_fc2 = layers.fully_connected(d_fc1, 240, activation_fn=None, scope='d_fc2')
   d_fc2 = lrelu(d_fc2)
   
   d_fc3 = layers.fully_connected(d_fc2, 1, activation_fn=tf.nn.sigmoid, scope='d_fc3')

   print 'x:',x
   print 'd_fc1:',d_fc1
   print 'd_fc2:',d_fc2
   print 'd_fc3:',d_fc3
   return d_fc3


def train(mnist_train):
   with tf.Graph().as_default():
     
      batch_size = 32

      # placeholder to keep track of the global step
      global_step = tf.Variable(0, trainable=False, name='global_step')
      
      # placeholder for mnist images
      images = tf.placeholder(tf.float32, [batch_size, 784], name='images')
      
      # placeholder for the latent z vector
      z = tf.placeholder(tf.float32, [batch_size, 100], name='z')

      # generate an image from noise prior z
      generated_images = G(z)

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
      G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=g_vars, global_step=global_step)
      D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=d_vars)

      saver = tf.train.Saver(max_to_keep=1)
   
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
      sess.run(init)

      # load previous checkpoint if there is one
      ckpt = tf.train.get_checkpoint_state('checkpoints/gan/')
      if ckpt and ckpt.model_checkpoint_path:
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model restored'
         except:
            print 'Could not restore model'
            pass


      # training loop
      step = sess.run(global_step)
      num_train = len(mnist_train)
      while True:
         s = time.time()
         step += 1

         epoch_num = step/(num_train/batch_size)

         # get random images from the training set
         batch_images = random.sample(mnist_train, batch_size)
         
         # generate z from a normal/uniform distribution between [-1, 1] of length 100
         batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)

         # run D
         sess.run(D_train_op, feed_dict={z:batch_z, images:batch_images})
         
         # run G
         sess.run(G_train_op, feed_dict={z:batch_z, images:batch_images})

         # get losses WITHOUT running the networks
         G_loss, D_loss = sess.run([errG, errD], feed_dict={z:batch_z, images:batch_images})
         
         while D_loss < 1e-4:
            sess.run(G_train_op, feed_dict={z:batch_z, images:batch_images})
            D_loss = sess.run([errD], feed_dict={z:batch_z, images:batch_images})
         
         if step%100==0:print 'epoch:',epoch_num,'step:',step,'G loss:',G_loss,' D loss:',D_loss,' time:',time.time()-s

         if step%2000 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, 'checkpoints/gan/checkpoint-', global_step=global_step)

            # generate some to write out
            batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
            gen_imgs = np.asarray(sess.run(generated_images, feed_dict={z:batch_z, images:batch_images}))
            random.shuffle(gen_imgs)
            # write out a few (10)
            c = 0
            for img in gen_imgs:
               img = np.reshape(img, [28, 28])
               plt.imsave('checkpoints/gan/images/0000'+str(step)+'_'+str(c)+'.png', img)
               if c == 5:
                  break
               c+=1


if __name__ == '__main__':

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir('checkpoints/gan/')
   except: pass
   try: os.mkdir('checkpoints/gan/images/')
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
      mnist_train.append(np.reshape(t, (784)))
   for t,l in zip(*val_set):
      mnist_train.append(np.reshape(t, (784)))
   for t,l in zip(*test_set):
      mnist_train.append(np.reshape(t, (784)))

   mnist_train = np.asarray(mnist_train)

   train(mnist_train)
