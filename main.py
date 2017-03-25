'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

from lib import generateGMMs
#from matplotlib.pylab import *

n_gauss = 4
n_latent = 16
n_hidden_units = 512
n_layer = 6
nb_epoch = 500

n_dim = 2
n_samples_train = 10000
n_samples_test = 1000

batch_size = 100
original_dim = n_dim
latent_dim = n_latent
intermediate_dim = n_hidden_units
epsilon_std = 1e-3


x = Input(batch_shape=(batch_size, original_dim))
h = x
for i in range(n_layer):
    h = Dropout(0.25)(Dense(intermediate_dim, activation='relu', W_regularizer='l1l2')(h))
z_mean = Dense(latent_dim, W_regularizer='l1l2')(h)
z_log_var = Dense(latent_dim, W_regularizer='l1l2')(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = []
for i in range(n_layer):
    decoder_h.append(Dense(intermediate_dim, activation='relu', W_regularizer='l1l2'))
decoder_mean = Dense(original_dim, activation='linear')
h_decoded = z
for i in range(n_layer):
    h_decoded = Dropout(0.25)(decoder_h[i](h_decoded))
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.mape(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# make training/testing datas
print('Generating data, takes around 25 secs')

np.random.seed(112)
GMMs_train = generateGMMs(n_gauss, n_dim, n_samples_train)
print('Done generating training set')
GMM = GMMs_train[0]
mean, var, weight, samples, pdf = GMM.mean, GMM.var, GMM.weight, GMM.samples, GMM.pdf
x_train = samples[:9000]
x_valid = samples[9000:]
'''
np.random.seed(113)
GMMs_test = generateGMMs(n_gauss, n_dim, n_samples_test)
print('Done generating testing set')
GMM = GMMs_test[0]
mean, var, weight, samples, pdf = GMM.mean, GMM.var, GMM.weight, GMM.samples, GMM.pdf
x_test = samples
'''

# train vae
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_valid, x_valid))

'''
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_valid_encoded = encoder.predict(x_valid, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1])
###plt.colorbar()
plt.show()
'''

#'''
# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_input
for i in range(n_layer):
    _h_decoded = decoder_h[i](_h_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

grid = np.random.multivariate_normal(np.zeros(latent_dim), np.eye(latent_dim), 1000)
x_gen = []

for z in grid:
    z_sample = np.array([z])
    x_decoded = generator.predict(z_sample)
    x_gen.append(x_decoded)

x_decoded = np.concatenate(x_gen)
plt.figure(figsize=(6, 6))
plt.scatter(x_decoded[:, 0], x_decoded[:, 1])
###plt.colorbar()
imgname = 'image\/' +str(n_gauss) + '_' +  str(n_hidden) + '*' + str(n_layer) + '+' + str(n_latent) + '_' + str(epsilon_std) + '.png'
plt.savefig(imgname)
#'''

'''
######
#
# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
x_gen = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        x_gen.append(x_decoded)

'''
