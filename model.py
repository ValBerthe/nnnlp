"""Model for the neural network."""
from classifier import classifier
import tensorflow as tf
import os
import time
import datetime
import numpy as np

dropout_keep_prob = 0.5


class model(object):
    """Specify the model class."""

    def __init__(self):
        """Define the init function."""
        self.glove = classifier()
        self.FILTER_SIZES = [3, 4, 5]
        self.NUM_FILTERS = 100
        self.EPOCHS = 40
        self.BATCH_SIZE = 100
        self.EVALUATE_EVERY = 100
        self.CHECKPOINT_EVERY = 100
        self.REG_LAMBDA = 0.05
        # Input layer.

        self.input_x = tf.placeholder(
            tf.float32, [None, None, self.glove.VEC_DIMENSIONS],
            name="input_x"
        )
        self.input_x = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(
            tf.float32, [None, len(self.glove.emotions)],
            name="input_y"
        )
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob"
        )

        # Blocks of convolution layers
        l2_loss = tf.constant(0.0)
        pooled_outputs = []
        for i, filter_size in enumerate(self.FILTER_SIZES):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [
                    filter_size,
                    self.glove.VEC_DIMENSIONS,
                    1,
                    self.NUM_FILTERS
                ]
                # Weights
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1),
                    name="W"
                )
                # Bias
                b = tf.Variable(tf.constant(
                    0.1,
                    shape=[self.NUM_FILTERS]),
                    name="b"
                )
                # Convolutional layer
                conv_layer = tf.nn.conv2d(
                    self.input_x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # ReLu / Non-linearity
                h = tf.nn.relu(
                    tf.nn.bias_add(conv_layer, b),
                    name="relu"
                )
                # Max-pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[
                        1,
                        self.glove.maxlen - filter_size + 1,
                        1,
                        1
                    ],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )
                pooled_outputs.append(pooled)

        n_filters_total = self.NUM_FILTERS * len(self.FILTER_SIZES)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, n_filters_total])
        print("TIBO INSHAPE::::::::::::::::::")
        print(self.h_pool_flat.shape)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat,
                self.dropout_keep_prob
            )

        # Output / predictions and scores
        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [n_filters_total, len(self.glove.emotions)],
                    stddev=0.1
                ), name="W")
            b = tf.Variable(
                tf.constant(0.1, shape=[len(self.glove.emotions)]),
                name="b"
            )
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores,
                labels=self.input_y
            )
            self.loss = tf.reduce_mean(losses) + self.REG_LAMBDA * l2_loss

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1)
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"),
                name="accuracy"
            )


# Session and Graph definition
with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = model()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step
        )

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "runs", timestamp)
        )
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir,
            sess.graph_def
        )

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(
            dev_summary_dir, sess.graph_def
        )

        # Checkpointing
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        # Tensorflow assumes this directory already exists,
        # so we need to create it
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """Train a step."""
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([
                train_op,
                global_step,
                train_summary_op,
                cnn.loss,
                cnn.accuracy
            ], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy)
            )
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """Evaluate model on a dev set."""
            x_batch = np.expand_dims(x_batch, axis=3)
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1
            }
            step, summaries, loss, accuracy = sess.run([
                global_step,
                dev_summary_op,
                cnn.loss,
                cnn.accuracy
            ], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy
            ))
            if writer:
                writer.add_summary(summaries, step)

        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """Generate a batch iterator for a dataset."""
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(
                        np.arange(data_size)
                    )
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]

        # Generate batches
        batches = batch_iter(
            list(zip(
                cnn.glove.training_data,
                cnn.glove.training_classes
            )), cnn.BATCH_SIZE, cnn.EPOCHS)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.expand_dims(x_batch, axis=3)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % cnn.EVALUATE_EVERY == 0:
                print("\nEvaluation:")
                dev_step(
                    cnn.glove.test_data,
                    cnn.glove.test_classes,
                    writer=dev_summary_writer
                )
            if current_step % cnn.CHECKPOINT_EVERY == 0:
                path = saver.save(
                    sess, checkpoint_prefix, global_step=current_step
                )
                print("Saved model checkpoint to {}\n".format(path))
