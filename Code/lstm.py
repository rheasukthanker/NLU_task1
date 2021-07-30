import tensorflow as tf
import numpy as np


class LSTMModel:
    def __init__(self,
                 sentence_len,
                 cg_sentence_len,
                 vocab_size,
                 learning_rate,
                 embedding_size,
                 grad_clip,
                 state_size,
                 softmax_size,
                 embedding_trainable,
                 loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits,
                 initializer=tf.contrib.layers.xavier_initializer):
        # Stop the program if the condition is not true => state size always needs to be bigger than softmax size
        assert(state_size >= softmax_size)

        self.lengths_without_tags = tf.placeholder(tf.int32, None, name="lengths_without_tags")
        self.vocab_size = vocab_size
        down_project = state_size != softmax_size

        # Define and initialize embedding matrix
        self.embedding_matrix = tf.get_variable(
            'embedding',
            [self.vocab_size, embedding_size],
            initializer=initializer(),
            trainable=embedding_trainable,
            dtype=tf.float32)

        # Define down projection matrix before softmax if running experiment c
        W_down_project = tf.get_variable(
            'W_down_project',
            [state_size, softmax_size],
            initializer=initializer(),
            trainable=True,
            dtype=tf.float32)

        # Define and initialize Weigth matrix and Bias vector for softmax layer
        with tf.variable_scope('softmax'):
            W_softmax = tf.get_variable(
                'W',
                [softmax_size, self.vocab_size],
                initializer=initializer(),
                trainable=True,
                dtype=tf.float32)

        # Define placeholders for input words and output labels (next input word)
        self.X_forward_backward = tf.placeholder(tf.int32, [None, sentence_len], name='X_forward_backward')
        self.Y_forward_backward = tf.placeholder(tf.int32, [None, sentence_len - 1], name='Y_forward_backward')
        self.X_condgen = tf.placeholder(tf.int32, [None, cg_sentence_len], name='X_condgen')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        sentence_indices = tf.constant(np.arange(cg_sentence_len), dtype=tf.int32, name='sentence_indices')

        # Define basic LSTM cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size, initializer=initializer())
        init_c = tf.get_variable("c", initializer=tf.zeros([1, state_size]), trainable=True, dtype=tf.float32)
        init_h = tf.get_variable("h", initializer=tf.zeros([1, state_size]), trainable=True, dtype=tf.float32)
        init_c_tiled = tf.tile(init_c, [self.batch_size, 1])
        init_h_tiled = tf.tile(init_h, [self.batch_size, 1])
        state = tf.contrib.rnn.LSTMStateTuple(init_c_tiled, init_h_tiled)

        # Lookup embeddings for each word in input X
        embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.X_forward_backward)
        #Mask for loss
        mask = tf.sequence_mask(self.lengths_without_tags + 1, maxlen=sentence_len, dtype=tf.float32)
        # Unroll LSTM cells for the sentence lengths
        with tf.variable_scope('UnrolledGraph'):
            # Initialize logits list, prob distribution list, initial loss
            logit_list_forward_pass = []
            self.prob_list_forward_pass = []
            self.loss = 0.0

            # Attach same LSTM cell after the last one until max sentence length it reached
            for time_step in range(sentence_len - 1):
                num_nonzero_masked = tf.math.maximum(tf.reduce_sum(mask[:, time_step]), 1)
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                output, state = lstm_cell(embeddings[:, time_step], state)

                # Softmax input equal to output of lstm cell or down projected state if running experiment c
                softmax_input = output
                if down_project:
                    softmax_input = tf.matmul(output, W_down_project)

                # After every word compute the logits and probability for the next word
                logits = tf.matmul(softmax_input, W_softmax)
                logit_list_forward_pass.append(logits)
                self.prob_list_forward_pass.append(tf.nn.softmax(logits))

                # Compute loss after every word
                intermediate_loss = loss_fn(labels=self.Y_forward_backward[:, time_step], logits=logits)*mask[:, time_step] 
                self.loss += tf.reduce_sum(intermediate_loss) / num_nonzero_masked

            init_c_tiled = tf.tile(init_c, [self.batch_size, 1])
            init_h_tiled = tf.tile(init_h, [self.batch_size, 1])
            state = tf.contrib.rnn.LSTMStateTuple(init_c_tiled, init_h_tiled)
            # Initialize logits_list and continuation of sentence list
            self.continued_sent = [self.X_condgen[:, 0]]
            for time_step in range(1, cg_sentence_len):
                if time_step > 1:
                    tf.get_variable_scope().reuse_variables()

                input_word_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.continued_sent[-1])
                output, state = lstm_cell(input_word_embeddings, state)

                # Softmax input equal to output of lstm cell or down projected state if running experiment c
                softmax_input = output
                if down_project:
                    softmax_input = tf.matmul(output, W_down_project)

                # After every word compute the logits and probability for the next word
                logits = tf.matmul(softmax_input, W_softmax)

                next_words = tf.where(
                    tf.math.less_equal(sentence_indices[time_step], self.lengths_without_tags),
                    self.X_condgen[:, time_step],
                    tf.math.argmax(logits, axis=1, output_type=tf.int32)
                )
                self.continued_sent.append(next_words)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        grads_clipped, _ = tf.clip_by_global_norm(grads, grad_clip)
        self.train_step = optimizer.apply_gradients(zip(grads_clipped, vars))


    def eval(self, X, Y, sentence_lengths_without_tags, session):
        """
        Predict the probability of the next word given all the previous words
        in a batch of sentences, and evaluate the performance using the given
        ground truth values. More specifically, X is a
        [batch_size, sentence_len] tensor such that X[i, j] contains the
        vocabulary index of the jth word in the ith sentence.

        Parameters
        ----------
        X:  Numpy array of shape [batch_size, sentence_len] such that X[i, j]
            contains the vocabulary index of the jth word in the ith sentence.
        Y: Tensor of shape [batch_size, sentence_len - 1] such that Y[i, j] contains
           the vocabulary index of the (j+1)th word in the ith sentence. The
           last index value in the sentences are unused.

        session: Already entered tensorflow session object.

        sentence_lengths_without_tags: Tensor of shape [batch_size], containing the length of
                          untagged version of each sentence in the batch.

        Returns
        -------
        loss: A float value indicating the accumulated loss when predicting all
              of the sentences in the given batch.
        Y_pred:  Numpy array of shape [batch_size, sentence_len - 1, vocab_size]
            such that Y[i, j, k] contains the probability that (j+1)th word in
            the ith sentence is the word with vocabulary index k.
        """
        batch_size, out_sentence_len = X.shape
        Y_pred = np.empty((batch_size, out_sentence_len - 1, self.vocab_size))
        loss, prob_list = session.run(
            [self.loss, self.prob_list_forward_pass],
            feed_dict={
                self.X_forward_backward: X,
                self.Y_forward_backward: Y,
                self.batch_size: batch_size,
                self.lengths_without_tags: sentence_lengths_without_tags
            }
        )
        for j, prob_dist in enumerate(prob_list):
            Y_pred[:, j, :] = prob_dist
        return loss, Y_pred

    def predict(self, X, sentence_lengths_without_tags, session):
        """
        Predict the probability of the next word given all the previous words
        in a batch of sentences, and evaluate the performance using the given
        ground truth values. More specifically, X is a
        [batch_size, sentence_len] tensor such that X[i, j] contains the
        vocabulary index of the jth word in the ith sentence.

        Parameters
        ----------
        X:  Numpy array of shape [batch_size, sentence_len] such that X[i, j]
            contains the vocabulary index of the jth word in the ith sentence.

        session: Already entered tensorflow session object.

        sentence_lengths_without_tags: Tensor of shape [batch_size], containing the length of
                          untagged version of each sentence in the batch.

        Returns
        -------
        Y_pred:  Numpy array of shape [batch_size, sentence_len - 1, vocab_size]
            such that Y[i, j, k] contains the probability that (j+1)th word in
            the ith sentence is the word with vocabulary index k.
        """
        batch_size, out_sentence_len = X.shape
        Y_pred = np.empty((batch_size, out_sentence_len - 1, self.vocab_size))
        prob_list = session.run(
            self.prob_list_forward_pass,
            feed_dict={
                self.X_forward_backward: X,
                self.batch_size: batch_size,
                self.lengths_without_tags: sentence_lengths_without_tags
            }
        )
        for j, prob_dist in enumerate(prob_list):
            Y_pred[:, j, :] = prob_dist
        return Y_pred

    def perform_train_step(self, X, Y, sentence_lengths_without_tags, session):
        """
        Performs a single training step by predicting the outputs for X,
        computing the gradients according to the ground truth given in Y, and
        applying the gradients.

        Parameters
        ----------
        X: Tensor of shape [batch_size, sentence_len] such that X[i, j] contains
           the vocabulary index of the jth word in the ith sentence.

        Y: Tensor of shape [batch_size, sentence_len - 1] such that Y[i, j] contains
           the vocabulary index of the (j+1)th word in the ith sentence. The
           last index value in the sentences are unused.

        sentence_lengths_without_tags: Tensor of shape [batch_size], containing the length of
                          untagged version of each sentence in the batch.

        session: Already entered tensorflow session object.

        Returns
        -------
        loss: A float value indicating the accumulated loss when predicting all
              of the sentences in the given batch.
        """
        batch_size, _ = X.shape
        _, current_loss = session.run(
            [self.train_step, self.loss],
            feed_dict={
                self.X_forward_backward: X,
                self.Y_forward_backward: Y,
                self.batch_size: batch_size,
                self.lengths_without_tags: sentence_lengths_without_tags
            }
        )
        return current_loss

    def cond_gen(self, X, sentence_lengths_without_tags, session):
        """
        Given a batch of seed sentences given in X with lengths given in sentence_lengths_without_tags,
        this function conditionally generates from an already trained LSTM model by picking
        the argmax of the probability distribution at each step. For a given sentence, this
        conditional generation continues until self.cg_sentence_len. If the user wants to stop sentences
        after an <eos> token is encountered, they must perform an appropriate postprocessing step
        on the results of this method.

        Parameters
        ----------
        X:  Numpy array of shape [batch_size, sentence_len] such that X[i, j]
            contains the vocabulary index of the jth word in the ith sentence.

        sentence_lengths_without_tags: Tensor of shape [batch_size], containing the length of
                                       untagged version of each sentence in the batch.

        session: Already entered tensorflow session object.

        Returns
        -------
        continued_sent: A [batch_size, self.cg_sentence_len] tensor containing the continued sentences
                        using the sentence beginnings given in X
        """
        batch_size, out_sentence_len = X.shape
        continued_sent = session.run(
            [self.continued_sent],
            feed_dict={
                self.X_condgen: X,
                self.batch_size: batch_size,
                self.lengths_without_tags: sentence_lengths_without_tags,
            }
        )
        return np.squeeze(continued_sent).T
