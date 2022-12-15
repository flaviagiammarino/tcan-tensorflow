import tensorflow as tf

from tcan_tensorflow.activations.activations import entmax

class SparseAttention(tf.keras.layers.Layer):

    def __init__(self, alpha=1.5):

        '''
        Sparse Attention Layer.

        Parameters:
        __________________________________
        alpha: float.
            Entmax parameter.
        '''

        self.alpha = alpha
        self.n = None

        super(SparseAttention, self).__init__()

    def build(self, input_shape):

        # Extract the number of inputs.
        if self.n is None:
            self.n = len(input_shape)

    def call(self, inputs, return_attention_scores=False):

        '''
        Parameters:
        __________________________________
        inputs: list.
            List of the following tensors: query, value and key (optional).

        return_attention_scores: bool.
            Whether to return the attention weights in addition to the output.
        '''

        # Extract the query, value and key matrices.
        if self.n == 2:
            query, value = inputs
            key = value
        else:
            query, value, key = inputs

        # Calculate the attention scores.
        scores = tf.matmul(query, key, transpose_b=True)

        # Calculate the attention weights.
        weights = entmax(scores, alpha=self.alpha)

        # Calculate the context vector.
        outputs = tf.matmul(weights, value)

        if return_attention_scores:
            return outputs, weights

        else:
            return outputs
