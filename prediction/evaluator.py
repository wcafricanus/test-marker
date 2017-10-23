import numpy as np
import os
import prediction.network as network3
from prediction.network import Network, ReLU

# Testing accuracy on test_data
import theano
import theano.tensor as T
mini_batch_size = 1


class Evaluator(object):
    def __init__(self):
        self.i = T.lscalar()
        self.net = Network([
            network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                   filter_shape=(20, 1, 5, 5),
                                   poolsize=(2, 2),
                                   activation_fn=ReLU),
            network3.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                                   filter_shape=(40, 20, 5, 5),
                                   poolsize=(2, 2),
                                   activation_fn=ReLU),
            network3.FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100, activation_fn=ReLU),
            network3.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

        cwd = os.path.dirname(__file__)
        all_params = np.load(cwd + '/cnnparams.npy')
        j = 0
        for layer in self.net.layers:
            layer.w = all_params[j]
            j = j + 1
            layer.b = all_params[j]
            j = j + 1
        self.net.reinit()

    def evaluate(self, test_dict):
        result_dict = {}
        images_2d = [value[0] for value in test_dict.values()]
        images = []
        for x in images_2d:
            if x is not None:
                images.append(np.reshape(x, (784)))
            else:
                images.append(np.zeros((784), np.float32))
        images_shared = network3.convert_data_shared(images)
        test_mb_predictions = theano.function(
            [self.i], self.net.layers[-1].y_out,
            givens={
                self.net.x:
                    images_shared[self.i * self.net.mini_batch_size: (self.i + 1) * self.net.mini_batch_size]
            },
            on_unused_input='warn')
        for key in test_dict:
            if test_dict[key][0] is None:
                result = [-1]
            else:
                result = test_mb_predictions(key)
            result_dict[key] = str(result[0] == test_dict[key][1])
        return result_dict
