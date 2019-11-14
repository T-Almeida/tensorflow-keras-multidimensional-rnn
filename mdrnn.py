import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

class MultiDimensionalRNN(tf.keras.layers.Layer):

    def __init__(self,
                 cell,
                 inital_state=None,
                 **kwargs):

        # outter layer keep track of the inner layer
        self.cell = cell
        self.inital_state = inital_state
        if not type(self.inital_state) is np.ndarray:
            self.inital_state = np.array([self.inital_state], dtype=np.float32)

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.rows = int(input_shape[1])
        self.columns = int(input_shape[2])
        if len(input_shape) == 3:
            self.features = 1
        else:
            self.features = int(input_shape[3])

        super().build(input_shape)

    # Base LOOP
    @tf.function
    def dynamic_tf_loop(self, input_data):

        input_shape = K.shape(input_data)
        #print("[DYNAMIC LOOP] input shape", input_shape)

        if self.inital_state is None:
            initial_state = tf.zeros((input_shape[1], self.cell.state_size))  # default state
        else:
            initial_state = self.inital_state

        input_flat = tf.TensorArray(dtype=tf.float32, size=self.rows*self.columns)

        input_flat = input_flat.unstack(input_data)  # 1D data representation

        # each entry (2D matrix entry) of the TensorArray is compised by a vector [BATCH, FEATURES]

        ######
        # find the recursive states base on the one dimentional index
        ######
        def get_back_state(index, states, columns):
            if 0 >= index % columns:  # out of the matrix/terminal case
                return initial_state
            else:
                return states.read(index-1)

        def get_up_state(index, states, columns):
            if index < columns:  # out of the matrix/terminal case
                return initial_state
            else:
                return states.read(index-columns)

        def get_diagonal_state(index, states, columns):
            if 0 >= index % columns or index < columns:  # out of the matrix/terminal case
                return initial_state
            else:
                return states.read(index-columns-1)
        ######
        # END
        ######

        # flat matrix with all the hidden states
        # clear_after_read must be False, since some entries (e.g. diagonal) could read up on 3 times the same state value
        states = tf.TensorArray(dtype=tf.float32, size=self.rows*self.columns, clear_after_read=False)

        # sequential loop -> recursive loop
        #print("Loop Iterations", self.rows*self.columns)
        for i in tf.range(self.rows*self.columns):

            states = states.write(i,
                                  self.cell(input_flat.read(i),
                                                 [get_back_state(i, states, self.columns),
                                                 get_up_state(i, states, self.columns),
                                                 get_diagonal_state(i, states, self.columns)])
                                 )

        return states.stack()

    @tf.function
    def call(self, x):
        input_shape = K.shape(x)
        batch = input_shape[0]

        input_data = K.reshape(x, [batch, -1, self.features])
        input_data = tf.transpose(input_data, (1, 0, 2))

        states = self.dynamic_tf_loop(input_data)

        # return the last computed states, i.e. h_{M,N}
        return tf.reshape(states[-1], self.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.inital_state.shape[0])
