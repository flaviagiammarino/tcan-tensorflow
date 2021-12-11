#---------------------------------------------- Set Up ----------------------------------------------#
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Attention
warnings.filterwarnings('ignore')

from tcan_tensorflow.layers.layers import SparseAttention

# Parameters.
seed = 1
alpha = 1
samples = 8
q_dim = 12
v_dim = 6
m_dim = 3
epochs = 10
batch_size = 1

# Data.
np.random.seed(seed)
q = np.random.normal(0, 1, (samples, q_dim, m_dim))
v = np.random.normal(0, 1, (samples, v_dim, m_dim))
k = np.random.normal(0, 1, (samples, v_dim, m_dim))
y = np.random.normal(0, 1, samples)

#---------------------------------------- Test 1: Untrained models ----------------------------------#

# Untrained sparse attention.
tf.random.set_seed(seed)
q1 = Input(shape=(q_dim, m_dim))
v1 = Input(shape=(v_dim, m_dim))
k1 = Input(shape=(v_dim, m_dim))
o1, s1 = SparseAttention(alpha=alpha)(inputs=[q1, v1, k1], return_attention_scores=True)
m1 = Model([q1, v1, k1], [o1, s1])
y1, w1 = m1([q, v, k])

# Untrained attention.
tf.random.set_seed(seed)
q2 = Input(shape=(q_dim, m_dim))
v2 = Input(shape=(v_dim, m_dim))
k2 = Input(shape=(v_dim, m_dim))
o2, s2 = Attention()(inputs=[q2, v2, k2], return_attention_scores=True)
m2 = Model([q2, v2, k2], [o2, s2])
y2, w2 = m2([q, v, k])

# Outputs comparison, should be True when alpha = 1 and False otherwise.
print(np.isclose(y1.numpy(), y2.numpy()).sum() == np.prod([samples, q_dim, m_dim]))

# Scores comparison, should be True when alpha = 1 and False otherwise.
print(np.isclose(w1.numpy(), w2.numpy()).sum() == np.prod([samples, q_dim, v_dim]))

#---------------------------------------- Test 2: Trained models ----------------------------------#

# Trained sparse attention.
tf.random.set_seed(seed)
q1 = Input(shape=(q_dim, m_dim))
v1 = Input(shape=(v_dim, m_dim))
k1 = Input(shape=(v_dim, m_dim))
o1, s1 = SparseAttention(alpha=alpha)(inputs=[q1, v1, k1], return_attention_scores=True)
m1 = Model([q1, v1, k1], o1)
m1.compile(loss='mse', optimizer='adam')
hist1 = m1.fit([q, v, k], y, epochs=epochs, batch_size=batch_size, verbose=0)
pred1 = m1.predict([q, v, k])

# Trained attention.
tf.random.set_seed(seed)
q2 = Input(shape=(q_dim, m_dim))
v2 = Input(shape=(v_dim, m_dim))
k2 = Input(shape=(v_dim, m_dim))
o2, s2 = Attention()(inputs=[q2, v2, k2], return_attention_scores=True)
m2 = Model([q2, v2, k2], o2)
m2.compile(loss='mse', optimizer='adam')
hist2 = m2.fit([q, v, k], y, epochs=epochs, batch_size=batch_size, verbose=0)
pred2 = m2.predict([q, v, k])

# Predictions comparison, should be True when alpha = 1 and False otherwise.
print(np.isclose(pred1, pred2).sum() == np.prod([samples, q_dim, m_dim]))

# Loss comparison, should be True when alpha = 1 and False otherwise.
print(np.isclose(hist1.history['loss'], hist2.history['loss']).sum() == epochs)
