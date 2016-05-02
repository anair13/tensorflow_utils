from os import path as osp
import other_utils as ou
import tensorflow as tf

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

def get_bias(shape, name='b'):
  b = tf.constant(0.1, shape=shape)
  return tf.Variable(b, name=name)

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

def l2_loss(err, name=None):
  with tf.scope('L2Loss') as scope:
    pass


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


class TFNet(object):
  def __init__(self, modelName=None, logDir='tf_logs/',
          modelDir='tf_models/'):
    self.g_ = tf.Graph()
    self.lossCollection_ = 'losses'
    self.modelName_      = modelName
    self.logDir_         = logDir
    self.modelDir_       = modelDir
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

  def add_batch_norm(self, x, scope, movingAvgFraction=0.999, scale=False):
    shp = ip.get_shape()
    if len(shp)==2:
      nOp = shp[1]
    else:
      assert len(shp) == 4
      nOp = shp[3]
    with tf.variable_scope(scope):
      beta = tf.Variable(tf.constant(0.0, shape=[nOp]),
          name='beta', trainable=True)
      gamma = tf.Variable(tf.constant(1.0, shape=[nOp]),
          name='gamma', trainable=scale)
      #Compute the mean and variance of batch
      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
      #Compute the moving average of mean and variance
      ema = tf.train.ExponentialMovingAverage(decay=movingAvgFraction)
      ema_apply_op = ema.apply([batch_mean, batch_var])
      ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
      def mean_var_with_update():
          with tf.control_dependencies([ema_apply_op]):
              return tf.identity(batch_mean), tf.identity(batch_var)
      mean, var = control_flow_ops.cond(phase_train,
          mean_var_with_update,
          lambda: (ema_mean, ema_var))

      normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
          beta, gamma, 1e-5, affine)
    return normed 

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
