import tensorflow as tf

IMG_SIZE = (9,9,1)

IMG_H = IMG_SIZE[0]
IMG_W = IMG_SIZE[1]
IMG_C = IMG_SIZE[2]

smoothing = 0.

EXP_DIR = './saved_model'

class Model:
    def __init__(self):

        self.graph = tf.Graph()

    def build_graph(self):
        with self.graph.as_default():
            self.state = tf.placeholder(tf.float32,[None,IMG_H,IMG_W,IMG_C],name='state')
            self.cap_w = tf.placeholder(tf.float32,[None,1],name='cap_w')
            self.cap_b = tf.placeholder(tf.float32,[None,1],name='cap_b')
            self.label = tf.placeholder(tf.float32,[None,1],name='label')

            y = tf.layers.conv2d(self.state,16,(2,2),activation=tf.nn.relu)
            y = tf.layers.conv2d(y,16,(2,2),activation=tf.nn.relu)
            y = tf.layers.conv2d(y,16,(2,2),activation=tf.nn.relu)
            y = tf.layers.flatten(y)
            y = tf.concat([y,self.cap_w,self.cap_b],axis=1)
            y = tf.layers.dense(y,128,activation=tf.nn.relu)
            y = tf.layers.dense(y,1)

            self.logit = tf.identity(y,name='logit')
            self.prob = tf.sigmoid(y,name='prob')

            self.loss = tf.identity(tf.losses.sigmoid_cross_entropy(self.label,self.logit),name='loss')

            self.optimizer = tf.train.AdamOptimizer()
            self.train_step = self.optimizer.minimize(self.loss,name='train_step')
    #            g_step = tf.Variable(0,trainable=False) 
    #           lr_init = 0.1
    #          lr = tf.train.exponential_decay(lr_init,g_step,5000,0.5,staircase=True)
    #            self.optimizer = tf.train.RMSPropOptimizer(0.01)
    #            self.optimizer = tf.train.MomentumOptimizer(lr,0.9,use_nesterov=True)
    #            self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    def train(self,sess,batch):
        _dict = {self.state:batch[0], self.cap_w:batch[1], self.cap_b:batch[2], self.label:batch[3]}
        _loss, _ = self.sess.run([self.loss,self.train_step],feed_dict=_dict)
        return _loss

    def inference(self,sess,batch):
        _dict = {self.state:batch[0], self.cap_w:batch[1], self.cap_b:batch[2]}
        _logit = self.sess.run([self.logit],feed_dict=_dict)
        return _logit

    def save(self,sess):
        n=0
        while True:
            try:
                builder = tf.saved_model.builder.SavedModelBuilder(EXP_DIR+'_%d'%n)
                break
            except AssertionError:
                print 'WARNING: Model already exists'
            n += 1 
        builder.add_meta_graph_and_variables(sess,['graph'])
        builder.save()

    def load(self,sess):
        tf.saved_model.loader.load(sess,['graph'],EXP_DIR)

        self.state = self.graph.get_tensor_by_name('state:0')
        self.cap_w = self.graph.get_tensor_by_name('cap_w:0')
        self.cap_b = self.graph.get_tensor_by_name('cap_b:0')
        self.label = self.graph.get_tensor_by_name('label:0')

        self.logit = self.graph.get_tensor_by_name('logit:0')
        self.prob = self.graph.get_tensor_by_name('prob:0')
        self.loss = self.graph.get_tensor_by_name('loss:0')
        self.train_step = self.graph.get_operation_by_name('train_step')
