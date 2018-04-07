from __future__ import absolute_import, division, print_function
import os
import sys
import tensorflow as tf
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH="data/model.ckpt"
FROZEN_GRAPH_PATH="frozen_graph/export_dir"

def objective(y, pos_label):
    # prediction objective
    if y == pos_label:
        return 1
    else:
        return 0


def data_transformation(train_path, pos_label):
    trainData = np.genfromtxt(train_path, delimiter=',', dtype=int)
    records = trainData[:, 1:]
    labels = trainData[:, :1]
    transformedlabels = [objective(ele, pos_label) for ele in labels]
    return (records, transformedlabels)


def train_logistic_regression(X_train, y_train):
    tf.reset_default_graph()
    sess = tf.Session()
    
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]], name="pixels")
    y_labels = tf.placeholder(tf.int32, [None], name="labels")
    y = tf.one_hot(y_labels, depth=2)

    W = tf.Variable(tf.zeros([X_train.shape[1], 2]), name="weights")
    b = tf.Variable(tf.zeros([2]), name="biases")
    y_hat = tf.matmul(x, W) + b

    pred = tf.argmax(tf.nn.softmax(y_hat), 1, name="predict_class")  # Softmax

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)), tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        sess.run(train, feed_dict={x: X_train, y_labels: y_train})
        if i % 1000 == 0:
            print('Cost , Accuracy')
            print(sess.run(
                [loss, accuracy], feed_dict={
                    x: X_train,
                    y_labels: y_train
                }))
    return sess


if __name__ == "__main__":
    pos_label = 3
    train_path = os.path.join(cur_dir, "./train.data")
    (X_train, y_train) = data_transformation(train_path, pos_label)
    sess = train_logistic_regression(X_train, y_train)
    if len(sys.argv) > 1:
        if sys.argv[1] == '--checkpoint':
            # CHECKPOINT FILE FOR VARIABLES
            # NEEDS TO KNOW HOW TO CONSTRUCT THE GRAPH USING SAVED VARIABLES
            saver = tf.train.Saver()
            save_path = saver.save(sess, CHECKPOINT_PATH)
        else:
            print('invalid usage')
            print('usage: python train.py --checkpoint')
            exit(1)
    else:
        #default: MetaGraph
        builder = tf.saved_model.builder.SavedModelBuilder(FROZEN_GRAPH_PATH) 
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()
    
    sess.close()
