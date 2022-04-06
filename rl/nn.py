from keras import Model, Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from tensorflow import keras
import numpy as np
import tensorflow as tf


class LossHistory(Callback):

    def __init__(self):
        super().__init__()
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))


def configure_optimizer(optimizer, learning_rate):
    if optimizer == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)
    if optimizer == "rms_prop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if optimizer == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    if optimizer != "adam":
        print("Unknown optimizer: {}. Falling back to adam as the default optimizer.".format(optimizer))
    return keras.optimizers.Adam(learning_rate=learning_rate)


class ANET(Model):

    def __init__(self, input_shape, hidden_layers, output_nodes, optimizer, learning_rate, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [Dense(nodes, activation=activation if activation != "linear" else None) for nodes, activation in hidden_layers]
        layers.append(Dense(output_nodes, activation="softmax"))
        self.model = Sequential(layers)
        self.compile(
            optimizer=configure_optimizer(optimizer, learning_rate),
            loss=keras.losses.kl_divergence,
        )
        self.model_trained = False
        if weight_file is not None:
            self.load_weights(weight_file)
            self.model_trained = True

        # predict on a random sample to inform model of input size. Necessary to allow LiteModel to convert our model
        self.predict(np.random.random((1, *input_shape)))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def fit(self, **kwargs):
        res = super().fit(**kwargs)
        return res

    def predict(
            self,
            x,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
    ):
        return super().predict(
            x,
            batch_size,
            verbose,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing
        )

    def get_config(self):
        super().get_config()


class Critic(Model):

    def __init__(self, input_shape, hidden_layers, optimizer, learning_rate, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [Dense(nodes, activation=activation if activation != "linear" else None) for nodes, activation in hidden_layers]
        layers.append(Dense(1, activation="tanh"))
        self.model = Sequential(layers)
        self.compile(
            optimizer=configure_optimizer(optimizer, learning_rate),
            loss=keras.losses.mse
        )
        self.model_trained = False
        try:
            self.load_weights(weight_file)
            self.model_trained = True
        except Exception as e:
            print("Unable to load weight file", e)

        # predict on a random sample to inform model of input size. Necessary to allow LiteModel to convert our model
        self.predict(np.random.random((1, *input_shape)))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def fit(self, **kwargs):
        res = super().fit(**kwargs)
        return res

    def predict(
            self,
            x,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
    ):
        return super().predict(
            x,
            batch_size,
            verbose,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing
        )

    def get_config(self):
        super().get_config()


class LiteModel:

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
