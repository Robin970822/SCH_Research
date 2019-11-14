from sklearn.model_selection import train_test_split
from data import loadData, saveData
import tensorflow as tf
import config

seed = config.seed
F = loadData(filename='feature.npy')
labels = loadData(filename='labels.npy')
x_train, x_test, y_train, y_test = train_test_split(
    F, labels, test_size=0.1, random_state=seed)

# Visualize decoder setting
# Parameters
LR = 0.01
training_epoch = 400
batch_size = 128
display_step = 20

# Network Parameters
n_input = F.shape[1]

# hidden layer settings
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
n_hidden_3 = 8  # 3rd layer num features

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("int64", [None, ])


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),

    'classifier_h1': tf.Variable(tf.random_normal([n_hidden_3, 2])),
}
bias = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),

    'classifier_b1': tf.Variable(tf.random_normal([2])),
}


# Build the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   bias['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   bias['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   bias['encoder_b3']))
    return layer_3


# Build the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   bias['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   bias['decoder_b2']))
    return layer_2


# Build the classifier
def classifier(x):
    layer_1 = tf.nn.softmax(
        tf.add(tf.matmul(x, weights['classifier_h1']), bias['classifier_b1']))
    return layer_1


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
classifier_op = classifier(encoder_op)

# Prediction
y_pred = decoder_op
# Labels are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

clf_y_pred = classifier_op
clf_y_true = Y
loss_clf = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(clf_y_true,
                                                    depth=2) * tf.log(clf_y_pred), reduction_indices=[1]))
acc = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(clf_y_pred, 1), clf_y_true), tf.float32))
optimizer_clf = tf.train.AdamOptimizer(LR).minimize(loss_clf)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)
total_batch = int(len(x_train) / batch_size)

# Training cycle
for epoch in range(training_epoch * 2):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = x_train[i * batch_size: (
            i + 1) * batch_size] if i < total_batch else x_train[i * batch_size: len(x_train)]
        # Run
        _, c = sess.run([optimizer, loss], feed_dict={X: batch_xs})
    if epoch % display_step == 0:
        print("Iteration: %04d " % (epoch), "loss=", "{:.9f}".format(c))

print('Triang CLF')
for epoch in range(training_epoch):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = x_train[i * batch_size: (
            i + 1) * batch_size] if i < total_batch else x_train[i * batch_size: len(x_train)]
        batch_ys = y_train[i * batch_size: (
            i + 1) * batch_size] if i < total_batch else y_train[i * batch_size: len(y_train)]
        # Run
        _, c = sess.run([optimizer_clf, loss_clf],
                        feed_dict={X: batch_xs, Y: batch_ys})
    if epoch % display_step == 0:
        a = sess.run(acc, feed_dict={X: x_test, Y: y_test})
        mse = sess.run(loss, feed_dict={X: x_test})
        print("Iteration: %04d " % (epoch), "loss=",
              "{:.9f} acc {:.9f} decode loss {:.9f}".format(c, a, mse))

code = sess.run(encoder_op, feed_dict={X: F})
saveData(code, filename='code.npy')
