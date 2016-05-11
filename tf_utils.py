from os import path as osp
import other_utils as ou
import tensorflow as tf
from easydict import EasyDict as edict
import time
import numpy as np

##
#Get weights
def get_weights(shape, stddev=0.1, name='w', wd=None, lossCollection='losses'):
  '''
    stddev: stddev of init
    
  '''
  w = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=stddev),
       name=name)
  #w = tf.Variable(tf.random_normal(shape, mean=0, stddev=stddev),
  #     name=name)
  if wd is not None:
    weightDecay = tf.mul(tf.nn.l2_loss(w), wd, name='w_decay')
    tf.add_to_collection(lossCollection, weightDecay)
  return w

##
#Get bias
def get_bias(shape, name='b'):
  b = tf.constant(0.1, shape=shape)
  return tf.Variable(b, name=name)

##
#L1 loss
def l1_loss(tensor, weight=1.0, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.
  Args:
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.
  Returns:
    the L1 loss op.
  """
  with tf.op_scope([tensor], scope, 'L1Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    return loss 

##
#Not implemented
def l2_loss(err, name=None):
  with tf.scope('L2Loss') as scope:
    pass

##
#Apply batch norm to a layer
def apply_batch_norm( x, scopeName, movingAvgFraction=0.999,
       scale=False, phase='train'):
  assert phase in ['train', 'test']
  shp = x.get_shape()
  if len(shp)==2:
    nOp = shp[1]
  else:
    assert len(shp) == 4
    nOp = shp[3]
  with tf.variable_scope(scopeName):
    beta = tf.Variable(tf.constant(0.0, shape=[nOp]),
        name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[nOp]),
        name='gamma', trainable=scale)
    ema   = tf.train.ExponentialMovingAverage(decay=movingAvgFraction)
    batchMean, batchVar = tf.nn.moments(x,\
            range(len(shp)-1), name='moments')
    ema_apply_op = ema.apply([batchMean, batchVar])
    if phase == 'train':
      with tf.control_dependencies([ema_apply_op]):
        mean, var = tf.identity(batchMean), tf.identity(batchVar)
    else:
      mean = ema.trainer.average(batchMean)
      var  = ema.trainer.average(batchVar) 
      assert mean is not None
      assert var is not None    
    return tf.nn.batch_normalization(x, mean, var,
        beta, gamma, 1e-5, scale)

##
#Helper class for constructing networks
class TFNet(object):
  def __init__(self, modelName=None, logDir='tf_logs/',
          modelDir='tf_models/'):
    self.g_ = tf.Graph()
    self.lossCollection_ = 'losses'
    self.modelName_      = modelName
    self.logDir_         = logDir
    self.modelDir_       = modelDir
    if modelName is not None:
      self.logDir_   = osp.join(self.logDir_, modelName)
      self.modelDir_ = osp.join(self.modelDir_, modelName)
    ou.mkdir(self.logDir_)
    ou.mkdir(self.modelDir_) 
    self.summaryWriter_  = None

  def get_weights(self, scopeName, shape, stddev=0.005, wd=None):
    '''
      wd: weight decay
    '''
    assert len(shape) == 2 or len(shape)==4
    if len(shape) == 2:
      nIp, nOp = shape
    else:
      _, _, nIp, nOp = shape
    with tf.variable_scope(scopeName) as scope:
      w = get_weights(shape, stddev=stddev, name='w', wd=wd,\
       lossCollection=self.lossCollection_)
      if len(shape)==2:
        b = get_bias([1, nOp], 'b')
      else:
        b = get_bias([nOp], 'b')
    return w, b 
  
  def get_conv_layer(self, scopeName, ip, shape, stride, padding='VALID',
             use_cudnn_on_gpu=None, stddev=0.005, wd=None):
    '''
      ip       : input variable
      scopeName: the scope in which the variable is declared
      shape    : the shape of the filter (same format as below)
      stride   : (h_stride, w_stride)
      padding  : "SAME", "VALID" 
               @cesarsalgado: https://github.com/tensorflow/tensorflow/issues/196
                 'SAME': Round up (partial windows are included)
                 'VALID': Round down (only full size windows are considered)
      tf.nn.conv2d
        input_tensor: [batch, height, width, channels]
        filter      : [height, width, in_channels, out_channels]
    '''
    kh, kw, nIp, nOp = shape
    with tf.variable_scope(scopeName) as scope:
      w = get_weights(shape, stddev=stddev, name='w', wd=wd,\
           lossCollection=self.lossCollection_)
      b = get_bias([nOp], 'b')
    conv = tf.nn.bias_add(tf.nn.conv2d(ip, w, 
             [1, stride[0], stride[1], 1], 
             padding, use_cudnn_on_gpu=use_cudnn_on_gpu), 
             b, name=scopeName) 
    return self.get_conv_layer_from_wb(ip, scopeName, w, b, stride, padding=padding,
             use_cudnn_on_gpu=use_cudnn_on_gpu)

  def get_conv_layer_from_wb(self, scopeName, ip, w, b, stride, padding='VALID', 
            use_cudnn_on_gpu=None):
    conv = tf.nn.bias_add(tf.nn.conv2d(ip, w, 
             [1, stride[0], stride[1], 1], 
             padding, use_cudnn_on_gpu=use_cudnn_on_gpu),
             b, name=scopeName)
    return conv

  def add_to_losses(self, loss):
    tf.add_to_collection(self.lossCollection_, loss) 

  def get_loss_collection(self):
    return tf.get_collection(self.lossCollection_)

  def get_total_loss(self):
    return tf.add_n(self.get_loss_collection(), 'total_loss')

  #Storing the losses 
  def add_loss_summaries(self):
    losses = self.get_loss_collection()
    for l in losses:
      tf.scalar_summary(l.op.name, l) 
 
  #Store the summary of all the trainable params
  def add_param_summaries(self):
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)

  #Store the summary of gradients
  def add_grad_summaries(self, grads):
    if grads is None:
      return
    for grad, var in grads:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

  ##
  #Start the logging of loss, gradients and parameters
  def init_logging(self, grads=None):
    
    self.add_loss_summaries()
    self.add_param_summaries()
    self.add_grad_summaries(grads)
    #Merge all summaries
    self.summaryOp_ = tf.merge_all_summaries()
    #Create a saver for saving all the model files   
    #max_to_keep: number of checkpoints to save
    self.saver_     = tf.train.Saver(tf.all_variables(), max_to_keep=5,
                       name=self.modelName_)

  #save the summaries
  def save_summary(self, smmry, sess, step):
    '''
      smmry: the result of evaluating self.summaryOp_
      step : the step number in the optimization
    '''
    if self.summaryWriter_ is None:
      self.summaryWriter_ = tf.train.SummaryWriter(self.logDir_,\
                 sess.graph)
    if not type(smmry) is list:
      smmry = [smmry]
    for sm in smmry:
      self.summaryWriter_.add_summary(sm, step)

  #save the model
  def save_model(self, sess, step):
    svPath = osp.join(self.modelDir_, 'model') 
    ou.mkdir(svPath) 
    self.saver_.save(sess, svPath, global_step=step)


##
#Helper class for easily training TFNets
class TFTrain(object):
  def __init__(self, ipVar, tfNet, solverType='adam', initLr=1e-3, 
        maxIter=100000, dispIter=1000, logIter=1000, batchSz=128):
    #input variables
    assert type(ipVar) is list
    self.ips_ = ipVar
    #net to be trained
    self.tfNet_    = tfNet
    self.maxIter_  = maxIter
    self.dispIter_ = dispIter 
    self.logIter_  = logIter
    self.batchSz_  = batchSz
 
    #initialize the step
    self.iter_  = tf.Variable(0, name='iteration')
  
    #define the solver
    if solverType == 'adam':
      self.opt_ = tf.train.AdamOptimizer(initLr)
    else:
      raise Exception('Solver not recognized')
  
    #the loss to be optimized
    self.loss_  = tfNet.get_total_loss()

    #gradient computation
    self.grads_ = self.opt_.compute_gradients(self.loss_)
    apply_gradient_op = self.opt_.apply_gradients(self.grads_, global_step=self.iter_)
    with tf.control_dependencies([apply_gradient_op]):
      self.train_op_ = tf.no_op(name='train')

    #init logging of gradients
    tfNet.init_logging(self.grads_)
    
    #keep track of time in training the net
    self.trTime_ = 0

  ##
  #
  def reset_train_time(self):
    self.trTime_ = 0

  ##
  #add the training accuracy/loss measure
  def add_loss_summaries(self, lossOps, lossNames=None):
    '''
     lossOps: the operator that stores which accuracies/losses
             need to logged 
    '''
    if not type(lossOps) == list:
      lossOps   = [lossOps]
    if lossNames is None:
      lossNames = ['smmry_%s' % l.name for l in lossOps]
    self.lossSmmry_ = edict()
    self.lossNames_ = edict()
    self.lossSmmry_['train'] = []
    self.lossSmmry_['val']   = []
    self.lossNames_['train'] = []
    self.lossNames_['val']   = []
    for l, n in zip(lossOps, lossNames):
      for tv in ['train', 'val']:
        #Train/Val summaries should not be merged with the other summaries
        name = '%s_%s' % (tv, n)
        self.lossNames_[tv].append(name)
        self.lossSmmry_[tv].append(tf.scalar_summary(name, l))
    self.lossOps_ = lossOps 

  ##
  #step the network by 1
  def step_by_1(self, sess, feed_dict, evalOps=[], isTrain=True):
    '''
      feed_dict: the input to the net
      evalOps  : the operators to be evaluated
    '''
    tSt = time.time()
    assert type(evalOps) == list
    if isTrain:
      ops = sess.run([self.train_op_, self.loss_] +  evalOps, feed_dict=feed_dict)
      ops = ops[1:]
    else:
      ops = sess.run([self.loss_] +  evalOps, feed_dict=feed_dict)
    self.trTime_ += (time.time() - tSt)
    return ops
  
  def print_display_str(self, step, lossNames, losses, isTrain=True):
    if not list(losses):
      losses = [losses]
      lossNames = [lossNames]
    if isTrain:
      T = self.trTime_
      self.reset_train_time()
      lossStr = 'Iter: %d, time for %d iters: %f \n ' % (step, self.dispIter_, T)
    else:
      lossStr = ''
    lossStr = lossStr + ''.join('%s: %.3f\t' % (n, l) for n,l in zip(lossNames, losses))
    print (lossStr)
 
  ##
  #train the net 
  def train(self, train_data_fn, val_data_fn, trainArgs=[], valArgs=[]):
    '''
      train_data_fn: returns feed_dict for train data
      val_data_fn  : returns feed_dict for val data
    '''
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      self.reset_train_time()
      
      #Start the iterations
      for i in range(self.maxIter_):
        #Fetch the training data
        trainDat = train_data_fn(self.ips_, self.batchSz_, *trainArgs)

        if np.mod(i, self.logIter_)==0:
          #evaluate the training losses and summaries
          N       = len(self.lossOps_)
          evalOps = self.lossOps_ + self.lossSmmry_['train'] + [self.tfNet_.summaryOp_]
          res     = self.step_by_1(sess, trainDat, evalOps = evalOps)
          ovLoss   = res[0]
          trainLosses = res[1:N+1]          
          #Save the summaries
          self.tfNet_.save_summary(res[N+1:], sess, i)
          #snapshot the model
          self.tfNet_.save_model(sess, i)
          self.print_display_str(i, self.lossNames_['train'], trainLosses)     
 
          #evaluate the validation losses and summaries 
          valDat  = val_data_fn(self.ips_, self.batchSz_, *valArgs)
          evalOps = self.lossOps_ + self.lossSmmry_['val']
          res     = self.step_by_1(sess, valDat, evalOps = evalOps, isTrain=False)
          ovValLoss = res[0]
          valLosses = res[1:N+1]          
          #Save the val summaries
          self.tfNet_.save_summary(res[N+1:], sess, i)
          self.print_display_str(i, self.lossNames_['val'], valLosses, False)     
        else: 
          ops    = self.step_by_1(sess, trainDat)
          ovLoss = ops[0]
        assert not np.isnan(ovLoss), 'Model diverged, NaN loss'
        assert not np.isinf(ovLoss), 'Model diverged, inf loss'
