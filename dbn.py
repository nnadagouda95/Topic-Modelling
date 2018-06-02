import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from scipy.io import loadmat
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class RSM():
    def __init__(self,
                 n_hidden_units=100,
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate=1e-3,
                 n_epochs=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 verbose=True):
        self.n_hidden_units = n_hidden_units
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.verbose = verbose

    def train(self, X):

        self.n_visible_units = X.shape[1]

        # Initialize RSM parameters
        self._build_model()

        sess.run(tf.variables_initializer([self.W, self.h, self.v]))

        if self.optimization_algorithm == 'sgd':
            self._train(X)
        else:
            raise ValueError("Invalid optimization algorithm.")
        return

    def _initialize_weights(self, weights):
        if weights:
            for name, value in weights.items():
                self.__setattr__(name, tf.Variable(value))
        else:
            stddev = 1.0 / np.sqrt(self.n_visible_units)

            self.W = tf.Variable(tf.random_normal([self.n_hidden_units,
                                                   self.n_visible_units],
                                                  stddev))
            self.h = tf.Variable(tf.random_normal(
                [self.n_hidden_units], stddev))
            self.v = tf.Variable(tf.random_normal(
                [self.n_visible_units], stddev))


    def _build_model(self, weights=None):

        self._initialize_weights(weights)

        # TensorFlow operations
        # with tf.device('/gpu:0'):
        self.visible_units_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                           self.n_visible_units])
        D = tf.squeeze(
            tf.reduce_sum(self.visible_units_placeholder, 1, keepdims=True))
        D = tf.expand_dims(D, 1)

        self.compute_hidden_units_op = tf.nn.sigmoid(
            tf.transpose(tf.matmul(self.W, tf.transpose(
                self.visible_units_placeholder)))
            + tf.matmul(D, tf.transpose(tf.expand_dims(self.h,1))))

        self.hidden_units_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                          self.n_hidden_units])

        self.compute_visible_units_op = tf.matmul(self.hidden_units_placeholder,
                                                  self.W) + self.v

        self.random_uniform_values = tf.Variable(
            tf.random_uniform([self.batch_size, self.n_hidden_units]))

        sample_hidden_units_op = tf.to_float(
            self.random_uniform_values < self.compute_hidden_units_op)

        self.random_variables = [self.random_uniform_values]

        # Positive gradient
        # Outer product. N is the batch size length.
        positive_gradient_op = tf.matmul(
            tf.expand_dims(self.compute_hidden_units_op, 2),
            tf.expand_dims(self.visible_units_placeholder, 1))

        # Negative gradient

        sample_hidden_units_gibbs_step_op = sample_hidden_units_op
        for t in range(self.contrastive_divergence_iter):
            sample_visible_units_op = self.visible_units_placeholder

            compute_visible_units_op = tf.matmul(self.compute_hidden_units_op,
            # compute_visible_units_op = tf.matmul(sample_hidden_units_gibbs_step_op,
                                                 self.W) + self.v

            softmax_op = tf.nn.softmax(compute_visible_units_op)

            sample_visible_units_op = tf.squeeze(tf.distributions.Multinomial(
                total_count=tf.squeeze(
                    tf.reduce_sum(sample_visible_units_op, 1, keep_dims=True)),
                probs=softmax_op).sample(1))

            D = tf.squeeze(tf.reduce_sum(sample_visible_units_op, 1, keepdims=True))
            D = tf.expand_dims(D, 1)

            compute_hidden_units_gibbs_step_op = tf.nn.sigmoid(
                tf.transpose(tf.matmul(self.W, tf.transpose(
                    sample_visible_units_op))) +
                tf.matmul(D, tf.transpose(tf.expand_dims(self.h,1))))

            random_uniform_values = tf.Variable(
                tf.random_uniform([self.batch_size, self.n_hidden_units]))

            sample_hidden_units_gibbs_step_op = tf.to_float(
                random_uniform_values < compute_hidden_units_gibbs_step_op)

            self.random_variables.append(random_uniform_values)

        negative_gradient_op = tf.matmul(
            tf.expand_dims(compute_hidden_units_gibbs_step_op, 2),
            tf.expand_dims(sample_visible_units_op, 1))

        compute_delta_W = tf.reduce_mean(
            positive_gradient_op - negative_gradient_op, 0)
        compute_delta_v = tf.reduce_mean(
            self.visible_units_placeholder - sample_visible_units_op, 0)
        compute_delta_h = tf.reduce_mean(
            sample_hidden_units_op - sample_hidden_units_gibbs_step_op, 0)

        self.update_W = tf.assign_add(self.W,
                                      self.learning_rate * compute_delta_W)
        self.update_v = tf.assign_add(self.v,
                                      self.learning_rate * compute_delta_v)
        self.update_h = tf.assign_add(self.h,
                                      self.learning_rate * compute_delta_h)

    def _train(self, _data):

        for iteration in range(1, self.n_epochs + 1):
            data = tf.data.Dataset.from_tensor_slices(_data)
            data = data.shuffle(100000)
            data = data.batch(self.batch_size)
            iter = data.make_one_shot_iterator()
            batch = iter.get_next()
            for _ in range(int(len(_data) / self.batch_size)):

                b = tf.Session().run(batch)
                sess.run(tf.variables_initializer(
                    self.random_variables))  # Need to re-sample from uniform distribution
                sess.run([self.update_W, self.update_v, self.update_h],
                         feed_dict={self.visible_units_placeholder: b})
            if self.verbose:
                error = self._compute_reconstruction_error(_data)
                print(">> Epoch %d finished \tRSM Reconstruction error %f" % (
                    iteration, error))

    def _compute_hidden_units(self, visible_units):

        return sess.run(self.compute_hidden_units_op,
                        feed_dict={
                            self.visible_units_placeholder: visible_units})

    def _compute_visible_units(self, hidden_units):

        return sess.run(self.compute_visible_units_op,
                        feed_dict={self.hidden_units_placeholder: hidden_units})

    def forward(self, X):

        output = self._compute_hidden_units(X)
        return output

    def _reconstruct(self, output):

        return self._compute_visible_units(output)

    def _compute_reconstruction_error(self, data):

        data_transformed = self.forward(data)
        data_reconstructed = self._reconstruct(data_transformed)

        return np.mean(np.sum((data_reconstructed - data) ** 2, 1))

    def save(self, save_path):
        import pickle

        with open(save_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @classmethod
    def load(cls, load_path):
        import pickle

        with open(load_path, 'rb') as fp:
            dct_to_load = pickle.load(fp)
            return cls.from_dict(dct_to_load)

    def to_dict(self):
        dct_to_save = {name: self.__getattribute__(name) for name in
                       self._get_param_names()}
        dct_to_save.update(
            {name: self.__getattribute__(name).eval(sess) for name in
             self._get_weight_variables_names()})
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):
        weights = {var_name: dct_to_load.pop(var_name) for var_name in
                   cls._get_weight_variables_names()}

        n_visible_units = dct_to_load.pop('n_visible_units')

        instance = cls(**dct_to_load)
        setattr(instance, 'n_visible_units', n_visible_units)

        # Initialize RBM parameters
        instance._build_model(weights)
        sess.run(tf.variables_initializer([getattr(instance, name) for name in
                                           cls._get_weight_variables_names()]))

        return instance

    @classmethod
    def _get_weight_variables_names(cls):
        return ['W', 'h', 'v']

    @classmethod
    def _get_param_names(cls):
        return ['n_hidden_units',
                'n_visible_units',
                'optimization_algorithm',
                'learning_rate',
                'n_epochs',
                'contrastive_divergence_iter',
                'batch_size',
                'verbose']


class RBM():
    def __init__(self,
                 n_hidden_units=100,
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate=1e-3,
                 n_epochs=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 verbose=True):
        self.n_hidden_units = n_hidden_units
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.verbose = verbose

    def train(self, X):

        self.n_visible_units = X.shape[1]

        # Initialize RBM parameters
        self._build_model()

        sess.run(tf.variables_initializer([self.W, self.h, self.v]))

        if self.optimization_algorithm == 'sgd':
            self._train(X)
        else:
            raise ValueError("Invalid optimization algorithm.")
        return

    def _initialize_weights(self, weights):
        if weights:
            for name, value in weights.items():
                self.__setattr__(name, tf.Variable(value))
        else:
            stddev = 1.0 / np.sqrt(self.n_visible_units)

            self.W = tf.Variable(tf.random_normal([self.n_hidden_units,
                                                   self.n_visible_units],
                                                  stddev))
            self.h = tf.Variable(tf.random_normal(
                [self.n_hidden_units], stddev))
            self.v = tf.Variable(tf.random_normal(
                [self.n_visible_units], stddev))


    def _build_model(self, weights=None):

        self._initialize_weights(weights)

        # TensorFlow operations
        with tf.device('/cpu:0'):
            self.visible_units_placeholder = tf.placeholder(tf.float32,
                                                            shape=[None,
                                                                   self.n_visible_units])
            self.compute_hidden_units_op = tf.nn.sigmoid(
                tf.transpose(tf.matmul(self.W, tf.transpose(
                    self.visible_units_placeholder))) + self.h)
            self.hidden_units_placeholder = tf.placeholder(tf.float32,
                                                           shape=[None,
                                                                  self.n_hidden_units])
            self.compute_visible_units_op = tf.nn.sigmoid(
                tf.matmul(self.hidden_units_placeholder, self.W) + self.v)
            self.random_uniform_values = tf.Variable(
                tf.random_uniform([self.batch_size, self.n_hidden_units]))
            sample_hidden_units_op = tf.to_float(
                self.random_uniform_values < self.compute_hidden_units_op)
            self.random_variables = [self.random_uniform_values]

            # Positive gradient
            # Outer product. N is the batch size length.
            positive_gradient_op = tf.matmul(
                tf.expand_dims(sample_hidden_units_op, 2),
                tf.expand_dims(self.visible_units_placeholder, 1))

            # Negative gradient

            sample_hidden_units_gibbs_step_op = sample_hidden_units_op
            for t in range(self.contrastive_divergence_iter):
                compute_visible_units_op = tf.nn.sigmoid(
                    tf.matmul(sample_hidden_units_gibbs_step_op,
                              self.W) + self.v)
                compute_hidden_units_gibbs_step_op = tf.nn.sigmoid(
                    tf.transpose(tf.matmul(self.W, tf.transpose(
                        compute_visible_units_op))) + self.h)
                random_uniform_values = tf.Variable(
                    tf.random_uniform([self.batch_size, self.n_hidden_units]))
                sample_hidden_units_gibbs_step_op = tf.to_float(
                    random_uniform_values < compute_hidden_units_gibbs_step_op)
                self.random_variables.append(random_uniform_values)

            negative_gradient_op = tf.matmul(
                tf.expand_dims(sample_hidden_units_gibbs_step_op, 2),
                tf.expand_dims(compute_visible_units_op, 1))

            compute_delta_W = tf.reduce_mean(
                positive_gradient_op - negative_gradient_op, 0)
            compute_delta_v = tf.reduce_mean(
                self.visible_units_placeholder - compute_visible_units_op, 0)
            compute_delta_h = tf.reduce_mean(
                sample_hidden_units_op - sample_hidden_units_gibbs_step_op, 0)

            self.update_W = tf.assign_add(self.W,
                                          self.learning_rate * compute_delta_W)
            self.update_v = tf.assign_add(self.v,
                                          self.learning_rate * compute_delta_v)
            self.update_h = tf.assign_add(self.h,
                                          self.learning_rate * compute_delta_h)

    def _train(self, _data):

        for iteration in range(1, self.n_epochs + 1):
            data = tf.data.Dataset.from_tensor_slices(_data)
            data = data.shuffle(100000)
            data = data.batch(self.batch_size)
            iter = data.make_one_shot_iterator()
            batch = iter.get_next()
            for _ in range(int(len(_data) / self.batch_size)):

                sess.run(tf.variables_initializer(
                    self.random_variables))  # Need to re-sample from uniform distribution
                sess.run([self.update_W, self.update_v, self.update_h],
                         feed_dict={
                             self.visible_units_placeholder: tf.Session().run(
                                 batch)})
            if self.verbose:
                error = self._compute_reconstruction_error(_data)
                print(">> Epoch %d finished \tRBM Reconstruction error %f" % (
                    iteration, error))

    def _compute_hidden_units(self, visible_units):

        return sess.run(self.compute_hidden_units_op,
                        feed_dict={
                            self.visible_units_placeholder: visible_units})

    def _compute_visible_units(self, hidden_units):

        return sess.run(self.compute_visible_units_op,
                        feed_dict={self.hidden_units_placeholder: hidden_units})

    def forward(self, X):

        output = self._compute_hidden_units(X)
        return output

    def _reconstruct(self, output):

        return self._compute_visible_units(output)

    def _compute_reconstruction_error(self, data):

        data_transformed = self.forward(data)
        data_reconstructed = self._reconstruct(data_transformed)
        return np.mean(np.sum((data_reconstructed - data) ** 2, 1))

    def save(self, save_path):
        import pickle

        with open(save_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @classmethod
    def load(cls, load_path):
        import pickle

        with open(load_path, 'rb') as fp:
            dct_to_load = pickle.load(fp)
            return cls.from_dict(dct_to_load)

    def to_dict(self):
        dct_to_save = {name: self.__getattribute__(name) for name in
                       self._get_param_names()}
        dct_to_save.update(
            {name: self.__getattribute__(name).eval(sess) for name in
             self._get_weight_variables_names()})
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):
        weights = {var_name: dct_to_load.pop(var_name) for var_name in
                   cls._get_weight_variables_names()}

        n_visible_units = dct_to_load.pop('n_visible_units')

        instance = cls(**dct_to_load)
        setattr(instance, 'n_visible_units', n_visible_units)

        # Initialize RBM parameters
        instance._build_model(weights)
        sess.run(tf.variables_initializer([getattr(instance, name) for name in
                                           cls._get_weight_variables_names()]))

        return instance

    @classmethod
    def _get_weight_variables_names(cls):
        return ['W', 'h', 'v']

    @classmethod
    def _get_param_names(cls):
        return ['n_hidden_units',
                'n_visible_units',
                'optimization_algorithm',
                'learning_rate',
                'n_epochs',
                'contrastive_divergence_iter',
                'batch_size',
                'verbose']


class DBN():

    def __init__(self,
                 hidden_layers_structure=[100, 100],
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate=1e-3,
                 n_epochs=10,
                 n_iter_backprop=100,
                 l2_regularization=1.0,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 dropout_p=0,
                 # float between 0 and 1. Fraction of the input units to drop
                 verbose=True):
        self.hidden_layers_structure = hidden_layers_structure
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.rbm_layers = None
        self.verbose = verbose
        self.n_iter_backprop = n_iter_backprop
        self.l2_regularization = l2_regularization
        self.dropout_p = dropout_p
        self.p = 1 - self.dropout_p
        self.rbm_class = RBM
        self.rsm_class = RSM
        self.oh_encoder = OneHotEncoder()

    def _initialize_weights(self, weights):
        if weights:
            for attr_name, value in weights.items():
                self.__setattr__(attr_name, tf.Variable(value))
        else:

            stddev = 1.0 / np.sqrt(self.input_units)
            self.W = tf.Variable(
                tf.random_normal([self.input_units, self.num_classes], stddev))

            self.b = tf.Variable(tf.random_normal([self.num_classes], stddev))

    def _build_model(self, weights=None):

        with tf.device('/cpu:0'):
            self.visible_units_placeholder = self.rbm_layers[
                0].visible_units_placeholder


            # Define tensorflow operation for a forward pass
            rbm_activation = self.visible_units_placeholder
            for i, rbm in enumerate(self.rbm_layers):
                if i==0:
                    D = tf.squeeze(
                        tf.reduce_sum(rbm_activation, 1, keepdims=True))
                    D = tf.expand_dims(D, 1)
                    # pos_hid_prob
                    rbm_activation = tf.nn.sigmoid(tf.transpose(
                        tf.matmul(rbm.W, tf.transpose(
                            rbm.visible_units_placeholder))) +
                                                   tf.matmul(D, tf.transpose(
                                                       tf.expand_dims(rbm.h,
                                                                      1))))
                else:
                    rbm_activation = tf.nn.sigmoid(
                                tf.transpose(
                                    tf.matmul(rbm.W, tf.transpose(rbm_activation))) + rbm.h)

            self.transform_op = rbm_activation
            self.input_units = self.rbm_layers[-1].n_hidden_units

            # weights and biases
            self._initialize_weights(weights)

            if self.optimization_algorithm == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)
            else:
                raise ValueError("Invalid optimization algorithm.")

            # operations
            self.y = tf.matmul(self.transform_op, self.W) + self.b
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes])
            self.output = tf.nn.softmax(self.y)
            self.cost_function = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y,
                                                           labels=self.y_))
            self.train_step = self.optimizer.minimize(self.cost_function)

    def train(self, X, y=None, pre_train=True):

        if pre_train:
            self._pre_train(X)

        if y is not None:
            self._fine_tuning(X, y)
        return

    def _pre_train(self, X):
        # Initialize rbm layers
        self.rbm_layers = list()
        for i, n_hidden_units in enumerate(self.hidden_layers_structure):
            if i == 0:
                rsm = self.rsm_class(n_hidden_units=n_hidden_units,
                                     activation_function=self.activation_function,
                                     optimization_algorithm=self.optimization_algorithm,
                                     learning_rate=self.learning_rate,
                                     n_epochs=self.n_epochs,
                                     contrastive_divergence_iter=self.contrastive_divergence_iter,
                                     batch_size=self.batch_size,
                                     verbose=self.verbose)

                self.rbm_layers.append(rsm)

            else:
                rbm = self.rbm_class(n_hidden_units=n_hidden_units,
                                     activation_function=self.activation_function,
                                     optimization_algorithm=self.optimization_algorithm,
                                     learning_rate=self.learning_rate,
                                     n_epochs=self.n_epochs,
                                     contrastive_divergence_iter=self.contrastive_divergence_iter,
                                     batch_size=self.batch_size,
                                     verbose=self.verbose)

                self.rbm_layers.append(rbm)

        input_data = X
        for rbm in self.rbm_layers:
            rbm.train(input_data)
            input_data = rbm.forward(input_data)

        return

    def _fine_tuning(self, data, _labels):
        self.num_classes = len(np.unique(_labels))

        self._build_model()
        sess.run(tf.variables_initializer([self.W, self.b]))

        labels = self.oh_encoder.fit_transform(_labels.reshape(-1, 1)).toarray()

        self._train(data, labels)

    def _train(self, _data, labels):
        for iteration in range(self.n_iter_backprop):
            data = tf.data.Dataset.from_tensor_slices((_data, labels))
            data = data.shuffle(100000)
            data = data.batch(self.batch_size)
            iter = data.make_one_shot_iterator()
            batch = iter.get_next()
            for _ in range(int(len(_data) / self.batch_size)):

                feed_dict = {
                    self.visible_units_placeholder: tf.Session().run(batch[0]),
                    self.y_: tf.Session().run(batch[1])}

                sess.run(self.train_step, feed_dict=feed_dict)

            if self.verbose:
                feed_dict = {self.visible_units_placeholder: _data,
                             self.y_: labels}

                error = sess.run(self.cost_function, feed_dict=feed_dict)
                print(">> Epoch %d finished \tANN training loss %f" % (
                iteration, error))

    def _forward(self, X, mode='train'):
        feed_dict = {self.visible_units_placeholder: X}

        if mode == 'train':
            return sess.run(self.transform_op,
                            feed_dict=feed_dict)
        elif mode == 'eval':
            return sess.run(self.output, feed_dict=feed_dict)

    def score(self, X,y):
        probs = self._forward(X, mode='eval')
        preds = np.argmax(probs, axis=1)
        print(
            'Accuracy is {}'.format(
                float(np.sum(preds == y)) / len(y)))

    def to_dict(self):
        dct_to_save = {name: self.__getattribute__(name) for name in
                       self._get_param_names()}
        dct_to_save.update(
            {name: self.__getattribute__(name).eval(sess) for name in
             self._get_weight_variables_names()})

        dct_to_save['num_classes'] = self.num_classes
        dct_to_save['rbm_layers'] = [rbm.to_dict() for rbm in self.rbm_layers]
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):

        weights = {var_name: dct_to_load.pop(var_name) for var_name in
                   cls._get_weight_variables_names()}

        num_classes = dct_to_load.pop('num_classes')

        instance = cls(**dct_to_load)

        setattr(instance, 'num_classes', num_classes)

        rbm_layers = dct_to_load.pop('rbm_layers')

        instance._build_model(weights)
        sess.run(tf.variables_initializer([getattr(instance, name) for name in
                                           cls._get_weight_variables_names()]))
        return instance

    def save(self, save_path):
        import pickle

        with open(save_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @classmethod
    def _get_param_names(cls):
        return ['hidden_layers_structure',
                'activation_function',
                'optimization_algorithm',
                'learning_rate_rbm',
                'n_epochs',
                'contrastive_divergence_iter',
                'batch_size',
                'verbose']

    @classmethod
    def _get_weight_variables_names(cls):
        return ['W', 'b']



if __name__ == '__main__':

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True))

    # change the file name
    train_data = loadmat('bbc_train_data.mat')

    data = train_data['data']
    labels = train_data['labels'].T

    x_train, x_test, Y_train, Y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=0)
    dbn = DBN(hidden_layers_structure=[500, 125, 64],
              activation_function='sigmoid',
              optimization_algorithm='sgd',
              learning_rate=1e-3,
              n_iter_backprop=20,
              n_epochs=20,
              contrastive_divergence_iter=1,
              batch_size=5)

    dbn.train(x_train, Y_train)

    preds = dbn.score(x_test, np.squeeze(Y_test))



