import tensorflow as tf 
import datetime
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import random
from tensorflow.keras import utils

def data_cifar10(flatten = True):
  
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

  if(flatten):
    X_train = X_train.reshape(-1, 32, 96, 1)
    X_test = X_test.reshape(-1, 32, 96, 1)
  else:
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_test = X_test.reshape(-1, 32, 32, 3)
 
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  
  X_train /= 255
  X_test /= 255

  y_train = utils.to_categorical(y_train, 10).astype(np.float32)
  y_test = utils.to_categorical(y_test, 10).astype(np.float32)

  return X_train, y_train, X_test, y_test

x_train, y_train, x_test, y_test = data_cifar10()
(x__train, y__train), (x__test, y__test) = tf.keras.datasets.cifar10.load_data()

batchSize = 256
epochs = 10
learningRate = 0.001

leaky_relu_alpha = 0.2
dropout_rate = 0.5

def conv2d( inputs , filters , stride_size ): #Слой для создания сверточного слоя
    out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] , padding="SAME" )
    return tf.nn.leaky_relu(out , alpha=leaky_relu_alpha )

def maxpool( inputs , pool_size , stride_size ): #Слой для применения maxpooling
    return tf.nn.max_pool2d(inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='VALID' , strides=[ 1 , stride_size , stride_size , 1 ] )

def dense(inputs , weights): #Слой для создания полносвязного слоя
    x = tf.nn.leaky_relu(inputs @ weights, alpha=leaky_relu_alpha )
    return tf.nn.dropout( x , rate=dropout_rate )

#Инициализатор переменных по форме
initializer = tf.initializers.glorot_uniform() 

#Функция для получения весов
def get_weight(  shape, name ): 
    return tf.Variable(initializer(shape) , name=name , trainable=True , dtype=tf.float32 )

output_classes = 10 

#Формы слоев
shapes = [
    [ 1 , 1 , 1 , 16 ] ,
    [ 2 , 2 , 16 , 16 ] ,
    [ 2 , 2 , 16, 32 ] ,
    [ 2 , 2 , 32 , 32 ] ,
    [ 2 , 2 , 32 , 64 ] ,
    [ 2 , 2 , 64 , 64 ] ,
    [ 3072 , 256 ] ,
    [ 256 , output_classes ] ,
]

#Создание весов
weights = []
for i in range(len(shapes)):
    weights.append( get_weight(shapes[i] , 'weight{}'.format( i )))

#Модель
def model( x ) :
    x = tf.cast( x , dtype=tf.float32 )
    c1 = conv2d( x , weights[0] , stride_size=1 )
    c1 = conv2d( c1 , weights[1] , stride_size=1 )
    p1 = maxpool( c1 , pool_size=2 , stride_size=2 )

    c2 = conv2d( p1 , weights[2] , stride_size=1 )
    c2 = conv2d( c2 , weights[3] , stride_size=1 )
    p2 = maxpool( c2 , pool_size=2 , stride_size=2 )

    c3 = conv2d( p2 , weights[4] , stride_size=1 )
    c3 = conv2d( c3 , weights[5] , stride_size=1 )
    p3 = maxpool( c3 , pool_size=2 , stride_size=2 )
    flatten = tf.reshape( p3 , shape=( tf.shape( p3 )[0] , -1 ))

    d1 = dense( flatten , weights[6])
    logits = tf.matmul( d1 , weights[7] )

    return tf.nn.softmax(logits)

#Функция подсчета ошибки
def loss( pred , target ): 
    return tf.losses.categorical_crossentropy( target , pred )

#Оптимизатор
optimizer = tf.optimizers.Adam(learningRate)

def train( model, inputs , outputs ):
    m = tf.keras.metrics.Accuracy() 
    with tf.GradientTape() as tape:
        current_loss = loss( model( inputs ), outputs)
    grads = tape.gradient( current_loss , weights )
    optimizer.apply_gradients( zip( grads , weights ) )
    m.update_state(np.argmax(outputs, axis=1), np.argmax(model(inputs), axis=1))
    return tf.reduce_mean(current_loss) , m.result().numpy()

path = r'C:\Users\Anastasiya\Python_tasks\tensorflow' #путь для сохранения даных для Tensorboard


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
loss_log_dir = path + '\cifar' + current_time + '\data'
loss_summary_writer = tf.summary.create_file_writer(loss_log_dir) #средство записи файла резюме для данного каталога

amount_batches = int(len(x_train) / batchSize) 

def print_log(current, amount, params):
  #формат прогрессбара 23/120 [=====>------------------------]

  bar_len = 50 
  percent = int(current * bar_len / amount)
  progressbar = ''

  for i in range(bar_len): 
    if(i < percent):
      progressbar += '='
    elif(i == percent):
      progressbar += '>'
    else:
      progressbar += '-'

  message = "\r" + str(current) + '/' + str(amount) + ' [' + progressbar + ']  '
  for key in params:
    message += key + str(params[key]) + '. '

  print(message, end='')


with loss_summary_writer.as_default(): 

  for epoch in range(1, epochs + 1):
    learningEpochStartTime = datetime.datetime.now() 
    print('Эпоха', epoch, '/', epochs)
    avg_loss = 0 
    for batch in range(0, len(x_train), batchSize):
      current_loss, accuracy = train( model , x_train[batch:batch + batchSize] , y_train[batch:batch + batchSize] )
      avg_loss += current_loss

      params = {'Длительность обучения на эпохе: ': datetime.datetime.now() - learningEpochStartTime,
                'loss: ': current_loss.numpy(), 
                'accuracy: ': accuracy} 
      if(batch >= len(x_train) - batchSize): 
        params['loss: '] = (avg_loss / amount_batches).numpy() 
      current_batch = int(batch / batchSize) + 1 
      print_log(current_batch, amount_batches, params)
    tf.summary.scalar("avg_loss", avg_loss, step=epoch) 
    tf.summary.scalar("accuracy", accuracy, step=epoch) 
    loss_summary_writer.flush()
    print() 
