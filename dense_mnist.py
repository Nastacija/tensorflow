import tensorflow as tf 
import datetime
import numpy as np 
from tensorflow.keras import utils 

def data_mnist(flatten = True):
  
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

  if(flatten):
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
  else:
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
 
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  
  X_train /= 255
  X_test /= 255

  y_train = utils.to_categorical(y_train, 10).astype(np.float32)
  y_test = utils.to_categorical(y_test, 10).astype(np.float32)

  return X_train, y_train, X_test, y_test

x_train, y_train, x_test, y_test = data_mnist()

#постоянные параметры
learningRate = 0.5
epochs = 10
batchSize = 100

#тренируемые параметры
trainableParams = [] 

trainableParams.append(tf.Variable(tf.random.normal([784, 200], stddev=0.03), name='W1'))
trainableParams.append(tf.Variable(tf.random.normal([200], stddev=0.03), name='b1'))
trainableParams.append(tf.Variable(tf.random.normal([200, 10], stddev=0.03), name='W2'))
trainableParams.append(tf.Variable(tf.random.normal([10]), name='b2'))

#полносвязный слой с несколькими функциями активации
def base(x , params):
    W1 = params[0]
    b1 = params[1]
    W2 = params[2]
    b2 = params[3]

    hiddenOut = tf.nn.relu(x@W1+b1)
    y = tf.nn.softmax(hiddenOut@W2+b2)
    return y

#модель
def model(x):
    y = base(x, trainableParams)
    return y

#функция подсчета ошибки
def loss(pred , target):
    return tf.losses.categorical_crossentropy(target , pred)

optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate) #оптимизатор
m = tf.keras.metrics.Accuracy() #метрика

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

def train(model, inputs, outputs): 

    with tf.GradientTape() as tape:
      current_loss = tf.reduce_mean(loss(model(inputs), outputs))

      grads = tape.gradient(current_loss , trainableParams)
      optimizer.apply_gradients(zip(grads , trainableParams)) #градиентный спуск
      m.update_state(np.argmax(outputs, axis=1), np.argmax(model(inputs), axis=1)) #подсчет точности сети
    
    return current_loss, m.result().numpy()

path = r'C:\Users\Anastasiya\Python_tasks\tensorflow' #путь для сохранения даных для Tensorboard


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
loss_log_dir = path + '\mnist' + current_time + '\data'
loss_summary_writer = tf.summary.create_file_writer(loss_log_dir) #средство записи файла резюме для данного каталога

amount_bathces = int(len(x_train) / batchSize) 

with loss_summary_writer.as_default(): 

  for epoch in range(1, epochs + 1): 
    learningEpochStartTime = datetime.datetime.now()
    print('Эпоха', epoch , '/', epochs)
    avg_loss = 0

    for batch in range(0, len(x_train), batchSize): 
      current_loss, accuracy = train(model, x_train[batch: batch + batchSize], y_train[batch: batch + batchSize])
      avg_loss += current_loss

      #параметры, которые будем выводить
      params = {'Время обучения на эпохе: ': datetime.datetime.now() - learningEpochStartTime,
                'loss: ': round(current_loss.numpy(), 4),
                'accuracy: ': round(accuracy, 4)} 
      if(batch >= len(x_train) - batchSize):
        params['loss: '] = round((avg_loss / amount_bathces).numpy(), 4)

      current_batch = int(batch / batchSize) + 1
      print_log(current_batch, amount_bathces, params)

    tf.summary.scalar("avg_loss", avg_loss, step=epoch) #Сохраняем данные для Tensorboard
    tf.summary.scalar("accuracy", accuracy, step=epoch) #Сохраняем данные для Tensorboard
    loss_summary_writer.flush()
    print() 

