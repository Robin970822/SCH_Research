from sklearn.svm import SVC


def get_model(input_shape, output_shape, model_type='MLP'):
    # MLP
    if model_type == 'MLP':
        import tensorflow as tf
        from tensorflow.keras.backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    elif model_type == 'SVM':
        model = SVC(kernel='linear', C=1)
    return model
