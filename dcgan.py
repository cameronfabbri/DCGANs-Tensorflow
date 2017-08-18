import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
import os
import requests
import gzip
import cPickle as pickle
import numpy as np

selu_ = 0
batch_size = 128

'''
   Batch normalization
   https://arxiv.org/abs/1502.03167
'''
def bn(x):
   return tf.layers.batch_normalization(x)

'''
   Self normalizing neural networks paper
   https://arxiv.org/pdf/1706.02515.pdf
'''
def selu(x):
   print 'Using SELU'
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def relu(x):
   return tf.nn.relu(x)

def G(z, batch_size):
   z = tf.layers.dense(z, 4*4*1024, name='g_z')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])

   conv1 = tf.layers.conv2d_transpose(z, 256, 5, strides=2, name='g_conv1', padding='SAME')
   if selu_: conv1 = selu(conv1)
   else: conv1 = relu(bn(conv1))

   conv2 = tf.layers.conv2d_transpose(conv1, 128, 5, strides=2, name='g_conv2', padding='SAME')
   if selu_: conv2 = selu(conv2)
   else: conv2 = relu(bn(conv2))
   
   conv3 = tf.layers.conv2d_transpose(conv2, 64, 5, strides=2, name='g_conv3', padding='SAME')
   if selu_: conv3 = selu(conv3)
   else: conv3 = relu(bn(conv3))

   conv4 = tf.layers.conv2d_transpose(conv3, 1, 5, strides=2, name='g_conv4', padding='SAME')
   conv4 = tf.nn.tanh(conv4)
   
   conv4 = conv4[:,:28,:28,:]
   return conv4

def D(x, reuse=False):

   conv1 = tf.layers.conv2d(x, 64, 5, strides=2, name='d_conv1', reuse=reuse, padding='SAME')
   conv1 = lrelu(conv1)
   
   conv2 = tf.layers.conv2d(conv1, 128, 5, strides=2, name='d_conv2', reuse=reuse, padding='SAME')
   conv2 = lrelu(bn(conv2))
   
   conv3 = tf.layers.conv2d(conv2, 256, 5, strides=2, name='d_conv3', reuse=reuse, padding='SAME')
   conv3 = lrelu(bn(conv3))

   conv5 = tf.reshape(conv3, [batch_size, -1])

   fc1 = tf.layers.dense(conv5, 1, name='d_fc1', reuse=reuse)
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
      D_fake = D(generated_images, reuse=True)

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
      summary_writer = tf.summary.FileWriter('checkpoints/dcgan/selu_'+str(selu_)+'/logs/', graph=tf.get_default_graph())

      # load previous checkpoint if there is one
      ckpt = tf.train.get_checkpoint_state('checkpoints/dcgan/selu_/'+str(selu_)+'/')
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

         # get losses WITHOUT running the networks
         G_loss, D_loss, summary = sess.run([errG, errD, merged_summary_op], feed_dict={z:batch_z, images:batch_images})
         summary_writer.add_summary(summary, step)
         
         if step%10==0:print 'epoch:',epoch_num,'step:',step,'G loss:',G_loss,' D loss:',D_loss,' time:',time.time()-s

         if step%100 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, 'checkpoints/dcgan/selu_'+str(selu_)+'/checkpoint-', global_step=global_step)

            # generate some to write out
            batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
            gen_imgs = np.asarray(sess.run(generated_images, feed_dict={z:batch_z, images:batch_images}))
            random.shuffle(gen_imgs)
            # write out a few (10)
            c = 0
            for img in gen_imgs:
               img = np.reshape(img, [28, 28])
               plt.imsave('checkpoints/dcgan/selu_'+str(selu_)+'/images/0000'+str(step)+'_'+str(c)+'.png', img, cmap=plt.cm.gray)
               if c == 5:
                  break
               c+=1


if __name__ == '__main__':

   try: os.makedirs('checkpoints/dcgan/selu_'+str(selu_)+'/logs/')
   except: pass
   try: os.makedirs('checkpoints/dcgan/selu_'+str(selu_)+'/images/')
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
