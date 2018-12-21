# coding: utf-8

# In[14]:


# File written by adc2181
# (I developed this in a jupyter notebook, hence the strange formatting)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


import spacy 
from spacy.tokenizer import Tokenizer
import numpy as np

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)

# Read the dict
def learn_pronunciations(augmented=False):
    if augmented: # Include beginning/middle/end indicators
        lang = 'data/local/lang/align_lexicon.txt'
        sep = '\t'
    else: 
        lang = './data/local/dict/lexicon.txt'
        sep = ' '
    
    lang = open(lang, 'r').readlines()
    
    pronounce = {}
    for line in lang:
        line = line.strip().split(sep)
        pronounce[line[0]] = line[1:]
    return pronounce

def pronounce_text(text):
    tokens = tokenizer(text)
#     return [phone for token in tokens for phone in pronounce.get(token.text)]
    pron = []
    for token in tokens:
        pronunciation = pronounce.get(token.text)
        if pronunciation is None:
            return None
        for phone in pronunciation:
            pron.append(phone)
    return pron        

def read_phones():
    phones = './data/local/lang/phone_map.txt'
    phones = open(phones, 'r').readlines()
    
    phone2id, id2phone = {}, {}
    for i, phone in enumerate(phones):
        phone = phone.split(' ')
        phone2id[phone[0]] = i
        
    for phone, i in phone2id.items():
        id2phone[i] = phone
    
    return phone2id, id2phone
        
def phonehot(phone):
    onehot = np.zeros(p_N)
    onehot[phone2id[phone]] = 1
    return onehot

def unphonehot(onehot):
    for i in onehot:
        if i == 1:
            return id2phone[i]
        
def phone_vector(text):
    pronunciation = pronounce_text(text)
    if pronunciation is None:
        return None
    
    vec = np.zeros(p_N)
    for phone in pronunciation:        
        vec += phonehot(phone)
        
    return vec

def phone_vec2bag(vec):
    phone_bag = []
    for i in range(p_N):
        if vec[i]:
            for j in range(int(vec[i])):
                phone_bag.append(id2phone[i])
    return phone_bag

phone2id, id2phone = read_phones()
p_N = len(phone2id)
print('p_N', p_N)
pronounce = learn_pronunciations()
pronunciation = pronounce_text('this is a sentence')
print('pronounced', pronunciation)    
vec = phone_vector('this is a sentence')
print('vectorized bag', vec)
print('unvectorized bag', phone_vec2bag(vec))


# In[16]:


import numpy as np
import cv2
import os
import imghdr

class ImageNet():
    
    def __init__(self, batch_size):
        
        self.file_dir = './data/imagenet/scaled'
        self.batch_size = batch_size
        
        self.files = []
        for file in os.listdir(self.file_dir):
            if file != '.':
                file = os.path.join(self.file_dir, file)
                self.files.append(file)
                
        self.images = []
        self.labels = []
        
        self.unpronouncable = 0
        
        for filename in self.files:
            assert(imghdr.what(filename) == 'jpeg')
            
            image = cv2.imread(filename)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            basename = os.path.basename(filename[1:].strip()).split('.')[0]
            vec = phone_vector(basename)
            
            if vec is None:
                self.unpronouncable += 1
            else:
                self.images.append(img_gray)
                self.labels.append(vec)
                # print(phone_vec2bag(vec))
            
        self.cur = 0
        self.N = len(self.images)
        
    def next_batch(self):
        batch = self.images[self.cur: self.cur + self.batch_size]
        batch_labels = self.labels[self.cur: self.cur + self.batch_size]
        self.cur += self.batch_size
        if self.cur > self.N: # Begin again
            self.cur = 0
        return np.array(batch), np.array(batch_labels)

print('Loading scaled ImageNet data')
imagenet = ImageNet(64)
print('N', imagenet.N)
print('unpronouncable', imagenet.unpronouncable)
batch, batch_label = imagenet.next_batch()
print('batch shape', batch.shape, batch_label.shape)


# In[17]:


tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32], name='X')
p_in = tf.placeholder(dtype=tf.float32, shape=[None, p_N], name='p_in')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 32 * 32])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2
print(inputs_decoder)

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


# In[18]:


def encoder(X_in, p_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 32, 32, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        
        p_in = tf.layers.dense(p_in, units=12, activation=lrelu)
        p = tf.layers.dense(p_in, units=n_latent, activation=lrelu)
        
        x = tf.concat([x, p], 1)
        
        mean = tf.layers.dense(x, units=n_latent)
        sd   = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mean + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mean, sd


# In[19]:


def decoder(sampled_z, p_in, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
       
        p_in = tf.layers.dense(p_in, units=12, activation=lrelu)
        p = tf.layers.dense(p_in, units=n_latent, activation=lrelu)
        x = tf.concat([x, p], 1)
    
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
        
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=32*32, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 32, 32])
        return img


# In[20]:


def discriminator(dec, keep_prob):
    with tf.variable_scope("discrim", reuse=None):
        activation = lrelu
        X = tf.reshape(dec, shape=[-1, 32, 32, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
    
        pred = tf.layers.dense(x, units=p_N)
        return pred


# In[21]:


sampled, mean, sd = encoder(X_in, p_in, keep_prob)
dec = decoder(sampled, p_in, keep_prob)
pred = discriminator(dec, keep_prob)


# In[22]:


unreshaped = tf.reshape(dec, [-1, 32*32])
img_loss = 0.00001 * tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
img_loss = tf.sqrt(img_loss)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mean) - tf.exp(2.0 * sd), 1)
latent_loss = tf.where(tf.is_nan(latent_loss), img_loss, latent_loss)
label_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=p_in, logits=pred))
loss = tf.reduce_mean(img_loss + label_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[23]:


imagenet = ImageNet(batch_size)

for i in range(imagenet.N * 3):
    batch, p_batch = imagenet.next_batch()
    sess.run(optimizer, feed_dict = {X_in: batch, p_in: p_batch, Y: batch, keep_prob: 0.8})
        
    if not i % 200:
        ls, d, i_ls, d_ls, l_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, label_loss, mean, sd], feed_dict = {X_in: batch, p_in: p_batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [32, 32]), cmap='gray')
        plt.show()
        print(phone_vec2bag(p_batch[0]))
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(l_ls), np.mean(d_ls))


# In[24]:


def rand_phones():
    vec = np.zeros(p_N)
    rands = np.random.randint(0, p_N, 3)
    for r in rands:
        vec[r] += 1
    return vec

randoms = [np.random.normal(0, 1, n_latent) for _ in range(20)]
p_rand = [rand_phones() for _ in range(20)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, p_in: p_rand, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [32, 32]) for i in range(len(imgs))]

for i, img in enumerate(imgs):
    print(phone_vec2bag(p_rand[i]))
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()

