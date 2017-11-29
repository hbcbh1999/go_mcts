import tensorflow as tf
import shutil
IMG_SIZE = (9,9,1)

IMG_H = IMG_SIZE[0]
IMG_W = IMG_SIZE[1]
IMG_C = IMG_SIZE[2]

smoothing = 0.

EXP_DIR = './saved_model/model'

def pipeline(states):

    def rot_states(states,k):
        return tf.map_fn(lambda state: tf.image.rot90(state,k), states)

    rot = [rot_states(states,i) for i in range(4)]

    flip_rot = [tf.reverse(s,axis=[1]) for s in rot]

    return rot+flip_rot

class Model:
    def __init__(self):

        self.graph = tf.Graph()

    def build_graph(self):
        self.graph.as_default()

        self.state = tf.placeholder(tf.float32,[None,IMG_H,IMG_W,IMG_C],name='state')
#        self.cap_w = tf.placeholder(tf.float32,[None,1],name='cap_w')
#        self.cap_b = tf.placeholder(tf.float32,[None,1],name='cap_b')
        self.label = tf.placeholder(tf.float32,[None,1],name='label')
        self.training = tf.placeholder(tf.bool,name='training')


        state_sym = pipeline(self.state)

        y = tf.layers.conv2d(self.state,filters=16,kernel_size=2,activation=tf.nn.relu,name='conv1')
        y = tf.layers.flatten(y)
        y = tf.layers.dense(y,64,activation=tf.nn.relu,name='dense1')
        y = tf.layers.dense(y,1,name='output')

        losses = []
        for states in state_sym:
            _y = tf.layers.conv2d(states,filters=16,kernel_size=2,activation=tf.nn.relu,name='conv1',reuse=True)
            _y = tf.layers.flatten(_y)
            _y = tf.layers.dense(_y,64,activation=tf.nn.relu,name='dense1',reuse=True)
            _y = tf.layers.dense(_y,1,name='output',reuse=True)
            _loss = tf.losses.sigmoid_cross_entropy(self.label,_y)
            losses.append(_loss)

        self.logit = tf.identity(y,name='logit')
        self.prob = tf.sigmoid(y,name='prob')

        #self.loss = tf.identity(tf.losses.sigmoid_cross_entropy(self.label,self.logit),name='loss')
        self.loss = tf.reduce_mean(tf.stack(losses),name='loss')

        self.optimizer = tf.train.AdamOptimizer()
#        self.optimizer = tf.train.RMSPropOptimizer(0.001)
    #    self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train_step = self.optimizer.minimize(self.loss,name='train_step')
    #            g_step = tf.Variable(0,trainable=False) 
    #           lr_init = 0.1
    #          lr = tf.train.exponential_decay(lr_init,g_step,5000,0.5,staircase=True)
    #            self.optimizer = tf.train.MomentumOptimizer(lr,0.9,use_nesterov=True)
    #            self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    def train(self,sess,batch):
#        _dict = {self.state:batch[0], self.cap_w:batch[1], self.cap_b:batch[2], self.label:batch[3]}
        _dict = {self.state:batch[0], self.label:batch[1], self.training:True}
        _loss, _ = sess.run([self.loss,self.train_step],feed_dict=_dict)
        return _loss

    def inference(self,sess,batch):
#        _dict = {self.state:batch[0], self.cap_w:batch[1], self.cap_b:batch[2]}
        _dict = {self.state:batch[0],self.training:False}
        with self.graph.as_default():
            _logit = sess.run([self.logit],feed_dict=_dict)
        return _logit

    def save(self,sess):
        while True:
            try:
                builder = tf.saved_model.builder.SavedModelBuilder(EXP_DIR)
                break
            except AssertionError:
                print 'WARNING: REMOVING PREVIOUS MODELS'
                shutil.rmtree(EXP_DIR)
        builder.add_meta_graph_and_variables(sess,['graph'])
        builder.save()

    def load(self,sess):
        try:
            tf.saved_model.loader.load(sess,['graph'],EXP_DIR)
            self.graph = tf.get_default_graph()
            self.state = self.graph.get_tensor_by_name('state:0')
#            self.cap_w = self.graph.get_tensor_by_name('cap_w:0')
#            self.cap_b = self.graph.get_tensor_by_name('cap_b:0')
            self.label = self.graph.get_tensor_by_name('label:0')
            self.training = self.graph.get_tensor_by_name('training:0')

            self.logit = self.graph.get_tensor_by_name('logit:0')
            self.prob = self.graph.get_tensor_by_name('prob:0')
            self.loss = self.graph.get_tensor_by_name('loss:0')
            self.train_step = self.graph.get_operation_by_name('train_step')
        except IOError:
            print 'Model does not exist, building a new one...'
            self.build_graph()
            sess.run(tf.global_variables_initializer())
