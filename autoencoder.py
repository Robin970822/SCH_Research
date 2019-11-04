import tensorflow as tf
from data import loadData, saveData

x_train = loadData(filename='x_train.npy')
y_train = loadData(filename='y_train.npy')

x_test = loadData(filename='x_test.npy')
y_test = loadData(filename='y_test.npy')

# Visualize decoder setting
# Parameters
LR = 0.01
training_epoch = 200
batch_size = 128
display_step = 1
example_to_show = 10

# Network Parameters
n_input = 4371  # MNIST data input (28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
n_hidden_3 = 128
n_hidden_4 = 128
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),

    # 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
bias = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}


# Build the encoder
def encoder(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                bias['encoder_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                bias['encoder_b2']))
    # layer_3 = tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
    #                             bias['encoder_b3']))
    # layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
    #                                bias['encoder_b4']))
    return layer_2


# Build the decoder
def decoder(x):
    # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
    #                                bias['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                   bias['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   bias['decoder_b3']))
    return layer_3


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Labels are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
total_batch = int(len(x_train) / batch_size)
# Training cycle
for phrase in range(10):
    print('Traing Autoencoder')
    for epoch in range(1000):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = x_train[i * batch_size: (
                i+1) * batch_size] if i < total_batch else x_train[i * batch_size: len(x_train)]
            # Run
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_xs})
        if epoch % 100 == 0:
            print("Iteration: %04d " % (epoch), "loss=", "{:.9f}".format(c))

print(sess.run(encoder_op, feed_dict={X: x_test})[0])
train_encode = sess.run(encoder_op, feed_dict={X: x_train})
test_encode = sess.run(encoder_op, feed_dict={X: x_test})
saveData(train_encode, filename='train_encode')
saveData(test_encode, filename='test_encode')
