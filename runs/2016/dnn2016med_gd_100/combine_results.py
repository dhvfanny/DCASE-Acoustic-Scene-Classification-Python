import csv
import math
import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pp


def get_results(file_name):
    # type: (string) -> list of tuples
    results = []
    if os.path.isfile(file_name):
        with open(file_name, 'rt') as f:
            for row in csv.reader(f, delimiter='\t'):
                results.append(row)
    else:
        raise IOError("Result file not found [%s]" % file_name)

    return results


def get_results_all_fold(path):
    tot_res = []
    for fold in xrange(1, 5):
        tot_res = get_results('result_fold' + str(fold) + '.csv')
    return tot_res


# Read First Results
mfcc_results = get_results_all_fold('mfcc_res.csv')

# Read Second Results
lfcc_results = get_results_all_fold('lfcc_res.csv')

# Read Third Results
antimfcc_results = get_results_all_fold('antimfcc_res.csv')

x_v = mfcc_results[:, 3]
y_v = lfcc_results[:, 3]
z_v = antimfcc_results[:, 3]
res_v = mfcc_results[:, 0][None]
lb = pp.LabelBinarizer()
res_v = lb.fit_transform(None)

x = tf.placeholder(tf.float32, shape=[None, 15])
y = tf.placeholder(tf.float32, shape=[None, 15])
z = tf.placeholder(tf.float32, shape=[None, 15])
w = tf.Variable([0.3, 0.3, 0.3], name="w")
res_model = (x * tf.exp(w[0]) + y * tf.exp(w[1]) + z * tf.exp(w[2])) / (tf.exp(w[0]) + tf.exp(w[1]) + tf.exp(w[2]))
res = tf.placeholder(tf.float32, shape=[None, 15])
error = tf.reduce_sum(tf.square(res - res_model))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
model = tf.initialize_all_variables()

# x_v = np.random.rand(15 * 78, 15)
# y_v = np.random.rand(15 * 78, 15)
# z_v = np.zeros((15 * 78, 15))
# res_v = np.random.rand(15 * 78, 15)

tot_error_v = 0.0
with tf.Session() as session:
    session.run(model)
    for i in range(len(x_v) / 30):
        x_batch = x_v[30 * i:30 * i + 29]
        y_batch = y_v[30 * i:30 * i + 29]
        z_batch = z_v[30 * i:30 * i + 29]
        res_batch = res_v[30 * i:30 * i + 29]
        _, error_v = session.run([train_op, error], feed_dict={x: x_batch,
                                                               y: y_batch,
                                                               z: z_batch,
                                                               res: res_batch})
        tot_error_v += error_v
    w_v = session.run(w)
sum_w_v = math.exp(w_v[0]) + math.exp(w_v[1]) + math.exp(w_v[2])
print("Model: {a:.3f}x + {b:.3f}y + {c:.3f}z".format(a=math.exp(w_v[0]) / sum_w_v, b=math.exp(w_v[1]) / sum_w_v,
                                                     c=math.exp(w_v[2]) / sum_w_v))
print("Error: {e:.3f}".format(e=tot_error_v))

min_error_v = 1000000000.0
mi = 1.0
mj = 1.0
for i in np.arange(0, 1, 0.01):
    for j in np.arange(0, 1 - i, 0.01):
        error_v = np.sum((i * x_v + j * y_v + (1 - i - j) * z_v - res_v) ** 2)
        if error_v < min_error_v:
            min_error_v = error_v
            mi = i
            mj = j

print("Model: {a:.3f}x + {b:.3f}y + {c:.3f}z".format(a=mi, b=mj, c=1 - mi - mj))
print("Error: {e:.3f}".format(e=min_error_v))
