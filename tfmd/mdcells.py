import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.keras.activations import sigmoid


class MultiDimensinalGRUCell(tf.keras.layers.Layer):
    def __init__(self, 
                 units,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        
        self.units = units
        self.state_size = units
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer) # TODO add to the weights
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer) # TODO add to the weights
        self.bias_regularizer = regularizers.get(bias_regularizer) # TODO add to the weights
        
        self.activation = activations.get(activation)
        
        super().__init__(**kwargs)

    def build(self, input_shape):
        
        #input shape [BATCH, S]
        recurrent_space_plus_feature_dim = self.state_size*3 + input_shape[-1]
        
        # Trainnable weights for the reset GATES
        self.w_rl = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_left_reset_weight')
        self.b_rl = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_left_reset_bias')
        
        self.w_rt = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_top_reset_weight')
        self.b_rt = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_top_reset_bias')
        
        self.w_rd = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_diagonal_reset_weight')
        self.b_rd = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_diagonal_reset_bias')
        
        # Trainnable weights for the feature (Z) GATES
        self.w_zi = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_input_feature_weight')
        self.b_zi = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_input_feature_bias')
        
        self.w_zl = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_left_feature_weight')
        self.b_zl = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_left_feature_bias')
        
        self.w_zt = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_top_feature_weight')
        self.b_zt = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_top_feature_bias')
        
        self.w_zd = self.add_weight(shape=(recurrent_space_plus_feature_dim, self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='recurrent_diagonal_feature_weight')
        self.b_zd = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='recurrent_diagonal_feature_bias')
        
        # projection weigts U
        self.u = self.add_weight(shape=(self.state_size*3, self.state_size),
                                      initializer=self.bias_initializer,
                                      name='U_weights')
        
        self.w_i = self.add_weight(shape=(input_shape[-1], self.state_size),
                                      initializer=self.recurrent_initializer,
                                      name='input_feature_weight')
        self.b_i = self.add_weight(shape=(self.state_size,),
                                      initializer=self.bias_initializer,
                                      name='input_feature_bias')
        
        super().build(input_shape)

    def call(self, x, states):
        
        # states: [BATCH, 3 (left, top, diagonal)]
        
        left_state = states[0] #(BATCH, RECURRENT_DIM)
        top_state = states[1] #(BATCH, RECURRENT_DIM)
        diagonal_state = states[2] #(BATCH, RECURRENT_DIM)
        
        q_vec = tf.concat([left_state, top_state, diagonal_state, x], axis=1) # [BATCH, 3*RECURRENT_DIM + INPUT_DIM]
        
                     # [BATCH, RECURRENT_DIM]  [1, RECURRENT_DIM]
        reset_left = K.bias_add(K.dot(q_vec, self.w_rl), self.b_rl)
        reset_left = sigmoid(reset_left) # [BATCH, RECURRENT_DIM]
        
        reset_top = K.bias_add(K.dot(q_vec, self.w_rt), self.b_rt)
        reset_top = sigmoid(reset_top) # [BATCH, RECURRENT_DIM]
        
        reset_diagonal = K.bias_add(K.dot(q_vec, self.w_rd), self.b_rd)
        reset_diagonal = sigmoid(reset_diagonal) # [BATCH, RECURRENT_DIM]

        reset = tf.concat([reset_left, reset_top, reset_diagonal], axis=1) # [BATCH, 3*RECURRENT_DIM]
        
        _z_input = K.bias_add(K.dot(q_vec, self.w_zi), self.b_zi) # [BATCH, RECURRENT_DIM]
        _z_left = K.bias_add(K.dot(q_vec, self.w_zl), self.b_zl) # [BATCH, RECURRENT_DIM]
        _z_top = K.bias_add(K.dot(q_vec, self.w_zt), self.b_zt) # [BATCH, RECURRENT_DIM]
        _z_diagonal = K.bias_add(K.dot(q_vec, self.w_zd), self.b_zd) # [BATCH, RECURRENT_DIM]
        
        _z_input = tf.expand_dims(_z_input, axis=-1)
        _z_left = tf.expand_dims(_z_left, axis=-1)
        _z_top = tf.expand_dims(_z_top, axis=-1)
        _z_diagonal = tf.expand_dims(_z_diagonal, axis=-1)
        
        _z = tf.concat([_z_input, _z_left, _z_top, _z_diagonal], axis=-1)
        _z = K.softmax(_z, axis=-1)
        
        # each will have dims # [BATCH, RECURRENT_DIM]
        z = tf.split(_z, num_or_size_splits=4, axis=-1) 
        z_input = K.squeeze(z[0], axis=-1)
        z_left = K.squeeze(z[1], axis=-1)
        z_top = K.squeeze(z[2], axis=-1)
        z_diagonal = K.squeeze(z[3], axis=-1)
        
        # compute candite hidden space
        _states = tf.concat([left_state, top_state, diagonal_state], axis=1) # [BATCH, 3*RECURRENT_DIM]
        reset_states = reset * _states # reset the hidden states # [BATCH, 3*RECURRENT_DIM]
        _h_reset_states = K.dot(reset_states, self.u) # [BATCH, RECURRENT_DIM]
        _h = K.bias_add(K.dot(x, self.w_i), self.b_i) # [BATCH, RECURRENT_DIM]
        _h = _h + _h_reset_states
        _h = self.activation(_h)

        h = z_left * left_state + z_top * top_state + z_diagonal * diagonal_state + z_input * _h

        # OUTPUT [BATCH, Features]
        return h#tf.random.normal((K.shape(x)[0], self.state_size))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.state_size)
 