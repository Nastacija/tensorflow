import tensorflow as tf
import numpy as np

# 1

np_3D = np.arange(12).reshape(3, 2, 2)
tsfl_3D = tf.convert_to_tensor(np_3D)

# 2

np1 = np.arange(6).reshape(2, 3)
tsr1 = tf.convert_to_tensor(np1)

np2 = np.arange(12).reshape(3, 4)
tsr2 = tf.convert_to_tensor(np2)

mult = 'tsr1*tsr2' #поэлементное перемножение векторов
#не работает, т.к. нужны тензоры одинаковой формы

matr = tf.matmul(tsr1, tsr2) #матричное умножение
#tf.Tensor(
#[[20 23 26 29]
 #[56 68 80 92]], shape=(2, 4), dtype=int32)

# 3

rgd = tf.ragged.constant([[5, 2, 6], [3, 4]])

where_rgd = 'rgd.device' #оборванный тензор не поддерживает этот метод

