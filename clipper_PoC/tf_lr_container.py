from __future__ import print_function
import os
import sys # to use this, you can use the base image(https://hub.docker.com/r/clipper/py-rpc/) to build your container
import rpc 
import numpy as np
import tensorflow as tf

class TFLRContainer(rpc.ModelContainerBase):
    def __init__(self, path):
        self.sess = tf.Session('', tf.Graph())
        with self.sess.graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(self.sess, path)

    def predict_ints(self, inputs):
        preds = self.sess.run('predict_class:0', feed_dict = {'pixels:0': inputs})
        return [str(pred) for pred in preds]


if __name__ == "__main__":
    print('Starting TensorFlow LR container')
    model_name = os.environ["CLIPPER_MODEL_NAME"]
    model_version = os.environ["CLIPPER_MODEL_VERSION"]
    ip = "127.0.0.1"
    port = 7000

    input_type = "integers"
    model_dir_path = os.environ["CLIPPER_MODEL_PATH"]
    model_files = os.listdir(model_dir_path)
    assert len(model_files) >= 2
    fname = os.path.splitext(model_files[0])[0]
    full_fname = os.path.join(model_dir_path, fname)
    print(full_fname)
    model = TFLRContainer(full_fname)

    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, port, model_name, model_version, input_type)
