import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import math

def regularnn(NHIDDEN=24, INPUTDIM=1, OUTPUTDIM=1, STDEV=0.5):
  x = tf.placeholder(dtype=tf.float32, shape=[None,INPUTDIM], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None,OUTPUTDIM], name="y")
  W = tf.Variable(tf.random_normal([INPUTDIM,NHIDDEN], stddev=STDEV, dtype=tf.float32))
  b = tf.Variable(tf.random_normal([NHIDDEN], stddev=STDEV, dtype=tf.float32))
  W_out = tf.Variable(tf.random_normal([NHIDDEN,OUTPUTDIM], stddev=STDEV, dtype=tf.float32))
  b_out = tf.Variable(tf.random_normal([OUTPUTDIM], stddev=STDEV, dtype=tf.float32))
  hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
  output = tf.matmul(hidden_layer,W_out) + b_out
  return x,y,output

def mdn(NHIDDEN=24, INPUTDIM=1, OUTPUTDIM=1, STDEV=0.5, KMIX=24):
  NOUT = KMIX * (2+OUTPUTDIM)
  x = tf.placeholder(dtype=tf.float32, shape=[None,INPUTDIM], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None,OUTPUTDIM], name="y")
  Wh = tf.Variable(tf.random_normal([INPUTDIM,NHIDDEN], stddev=STDEV, dtype=tf.float32))
  bh = tf.Variable(tf.random_normal([NHIDDEN], stddev=STDEV, dtype=tf.float32))
  Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
  bo = tf.Variable(tf.random_normal([NOUT], stddev=STDEV, dtype=tf.float32))
  hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
  output = tf.matmul(hidden_layer,Wo) + bo
  return x,y,output

def get_mixture_coef(output, KMIX=24, OUTPUTDIM=1):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX*OUTPUTDIM], name="mixparam")
  splits = tf.split(1, 2 + OUTPUTDIM, output)
  out_pi = splits[0]
  out_sigma = splits[1]
  out_mu = tf.pack(splits[2:], axis=2)
  out_mu = tf.transpose(out_mu, [1,0,2])
  # use softmax to normalize pi into prob distribution
  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.sub(out_pi, max_pi)
  out_pi = tf.exp(out_pi)
  normalize_pi = tf.inv(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.mul(normalize_pi, out_pi)
  # use exponential to make sure sigma is positive
  out_sigma = tf.exp(out_sigma)
  return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
  oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
  result = tf.sub(y, mu)
  result = tf.transpose(result, [2,1,0])
  result = tf.mul(result,tf.inv(sigma + 1e-8))
  result = -tf.square(result)/2
  result = tf.mul(tf.exp(result),tf.inv(sigma + 1e-8))*oneDivSqrtTwoPI
  result = tf.reduce_prod(result, reduction_indices=[0])
  return result

def get_lossfunc(out_pi, out_sigma, out_mu, y):
  result = tf_normal(y, out_mu, out_sigma)
  kernel = result
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  beforelog = result
  result = -tf.log(result + 1e-8)
  return tf.reduce_mean(result),kernel,beforelog

def generate_ensemble(out_pi, out_mu, out_sigma, x_test, M = 10, OUTPUTDIM=1):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M, OUTPUTDIM) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0
  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      for d in range(0, OUTPUTDIM):
        idx = np.random.choice(24, 1, p=out_pi[i])
        mu = out_mu[idx,i,d]
        std = out_sigma[i, idx]
        result[i, j, d] = mu + rn[i, j]*std
  return result

# 1d to 1d test case
def oned2oned():
  NSAMPLE = 250

  y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
  r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
  x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)

  x,y,output = mdn()

  out_pi, out_sigma, out_mu = get_mixture_coef(output)
  lossfunc,k,bl = get_lossfunc(out_pi, out_sigma, out_mu, y)
  train_op = tf.train.AdamOptimizer().minimize(lossfunc)

  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())

  plt.figure(figsize=(8, 8))
  plt.plot(x_data,y_data,'ro', alpha=0.3)
  plt.show()

  NEPOCH = 10000
  loss = np.zeros(NEPOCH) # store the training progress here.
  for i in range(NEPOCH):
    sess.run(train_op,feed_dict={x: x_data, y: y_data})
    loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})
    print(loss[i])

  plt.figure(figsize=(8, 8))
  plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
  plt.show()

  x_test = np.float32(np.arange(-15,15,0.1))
  NTEST = x_test.size
  x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

  out_pi_test, out_sigma_test, out_mu_test = sess.run(get_mixture_coef(output), feed_dict={x: x_test})

  y_test = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test, x_test, M=1)

  plt.figure(figsize=(8, 8))
  plt.plot(x_data,y_data,'ro', x_test,y_test[:,:,0],'bo',alpha=0.3)
  plt.show()

# 1d to 2d test case
def oned2twod():
  NSAMPLE = 250
  fig = plt.figure()
  ax = Axes3D(fig) 
  z_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
  r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
  x1_data = np.float32(np.sin(0.75*z_data)*7.0+z_data*0.5+r_data*1.0)
  x2_data = np.float32(np.sin(0.5*z_data)*7.0+z_data*0.5+r_data*1.0)

  ax.scatter(x1_data, x2_data, z_data)
  ax.legend()
  plt.show()

  x_data = np.dstack((x1_data,x2_data))

  x,y,output = mdn(INPUTDIM=1, OUTPUTDIM=2)
  out_pi, out_sigma, out_mu = get_mixture_coef(output, OUTPUTDIM=2)
  lossfunc,kernel,beforelog = get_lossfunc(out_pi, out_sigma, out_mu, y)

  train_op = tf.train.AdamOptimizer().minimize(lossfunc)

  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())

  NEPOCH = 10000
  loss = np.zeros(NEPOCH) # store the training progress here.
  for i in range(NEPOCH):
    sess.run(train_op,feed_dict={x: z_data, y: x_data[:,0,:]})
    loss[i] = sess.run(lossfunc, feed_dict={x: z_data, y: x_data[:,0,:]})
    print(str(i) + ":" + str(loss[i]))

    #loss[i],k,bl = sess.run([lossfunc,kernel,beforelog], feed_dict={x: z_data, y: x_data[:,0,:]})
    #print(str(i) + ":" + str(loss[i]) + "," + str(k) + "" + str(bl))

  plt.figure(figsize=(8, 8))
  plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
  plt.show()

  x_test = np.float32(np.arange(-10.5,10.5,0.1))
  NTEST = x_test.size
  x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

  out_pi_test, out_sigma_test, out_mu_test = sess.run(get_mixture_coef(output, OUTPUTDIM=2), feed_dict={x: x_test})

  y_test = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test, x_test, M=1,OUTPUTDIM=2)

  fig = plt.figure()
  ax = Axes3D(fig) 
  ax.scatter(y_test[:,0,0], y_test[:,0,1], x_test, c='r')
  ax.scatter(x1_data, x2_data, z_data, c='b')
  ax.legend()
  plt.show()

oned2oned()
oned2twod()
