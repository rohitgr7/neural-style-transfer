import numpy as np
import tensorflow as tf
from scipy import io


class VGG:

    def __init__(self, weights_path, pool_type='max', alpha=0.0):

        # Required VGG19 layers
        self.layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv_4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
        ]

        # Load the weights
        self.weights = io.loadmat(weights_path)['layers'][0]

        # Load the hyperparameters
        self.pool_type = pool_type
        self.alpha = alpha

        # Mean values of pixels for vgg with RGB channels
        self.mean_pixels = np.array(
            [123.68, 116.779, 103.939], dtype=np.float32)

    def preprocess_input(self, image):

        image = image - self.mean_pixels

        # Reshaping to (1, H, W, C)
        image = image.reshape((1,) + image.shape)

        # RGB to BGR
        return image[..., ::-1]

    def unpreprocess_input(self, image):

        # BGR to RGB
        image = image[..., ::-1]

        # Reshaping to (H, W, C)
        image = image.reshape(image.shape[1:])

        return image + self.mean_pixels

    def _conv_layer(self, input_data, weights, bias):

        # Create tensorflow instance of the weights
        weights = tf.constant(weights)
        bias = tf.constant(bias)

        conv = tf.nn.conv2d(
            input_data,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        return tf.nn.bias_add(conv, bias)

    def _max_pool_layer(self, input_data):

        return tf.nn.max_pool(
            input_data,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='VALID'
        )

    def _avg_pool_layer(self, input_data):
        return tf.nn.avg_pool(
            input_data,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='VALID'
        )

    def _relu_layer(self, input_data):
        return tf.nn.relu(input_data) - self.alpha * tf.nn.relu(-input_data)

    def forward(self, image, scope='temp'):

        with tf.variable_scope(scope):
            layers = {}
            current = image

            for i, layer in enumerate(self.layers):
                if layer[:4] == 'conv':
                    weights = self.weights[i][0][0][2][0][0]
                    bias = self.weights[i][0][0][2][0][1]

                    weights = np.transpose(weights, (1, 0, 2, 3))
                    bias = bias.reshape((-1))
                    current = self._conv_layer(current, weights, bias)

                elif layer[:4] == 'pool':
                    if self.pool_type == 'max':
                        current = self._max_pool_layer(current)
                    elif self.pool_type == 'avg':
                        current = self._avg_pool_layer(current)

                elif layer[:4] == 'relu':
                    current = self._relu_layer(current)

                layers[layer] = current

            return layers
