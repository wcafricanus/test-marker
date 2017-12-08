# encoding: UTF-8

import os

import numpy as np
import tensorflow as tf


class Evaluator(object):
    def __init__(self):
        cwd = os.path.dirname(__file__)
        tf.reset_default_graph()
        imported_meta = tf.train.import_meta_graph(cwd + "/trained-models/model_final.meta")

        self.sess = tf.Session()
        imported_meta.restore(self.sess, tf.train.latest_checkpoint(cwd + '/trained-models/'))

    def evaluate(self, test_dict):
        images_2d = [value[0].reshape(28, 28, 1) if value[0] is not None else np.zeros((28, 28, 1)) for value in
                     test_dict.values()]
        labels = [value[1] for value in
                     test_dict.values()]
        keys = [value for value in test_dict.keys()]
        result = {}

        Y = self.sess.graph.get_tensor_by_name('Y:0')

        input_data = {'X:0': images_2d, 'tst:0': True, 'pkeep:0': 1, 'pkeep_conv:0': 1.0}

        Y_output = self.sess.run(Y,input_data)
        predictions = np.argmax(Y_output, 1)
        print(predictions)
        for i in range(len(keys)):
            is_correct = predictions[i] == labels[i]
            result[keys[i]] = str(is_correct)
            # if predictions[i] != labels[i]:
            #     plt.figure(1)
            #     plt.subplot(211)
            #     plt.axis("off")
            #     plt.imshow(input_data['X:0'][i].reshape(28,28), cmap='gray')
            #     ax = plt.subplot(212)
            #     x_pos = np.arange(10)
            #     plt.bar(x_pos, Y_output[i])
            #     ax.xaxis.set_ticks(np.arange(0, 10, 1))
            #     ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1d'))
            #     plt.show()
        return result
