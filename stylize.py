import numpy as np
import tensorflow as tf


class Stylize:

    def __init__(self, net, content_image, style_images, init_gen_image, args):

        self.verbose = args.verbose

        self.net = net

        # Preprocess image
        self.C0 = self.net.preprocess_input(content_image)
        self.S0 = np.array([self.net.preprocess_input(st_image)
                            for st_image in style_images], dtype=np.float32)
        self.X0 = self.net.preprocess_input(init_gen_image)

        # Layers to calculate loss
        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
        self.content_losstype = args.content_losstype

        # Loss weights
        self.content_layer_weights = args.content_layer_weights
        self.style_layer_weights = args.style_layer_weights
        self.alpha = args.alpha
        self.beta = args.beta
        self.tv = args.tv

        # Optimization
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.num_iter = args.num_iter
        self.print_iter = args.print_iter

        if self.verbose:
            print('Parameters initialized....')

    def transfer_style(self):

        # Buid graph
        self._build_graph()

        if self.verbose:
            print('Graph build complete....')
            print('Creating session....')

        # Session creation
        with tf.Session() as sess:

            # Get the required optimizer
            optimizer = self._get_optimizer()
            # Values for the placeholders
            feed_dict = {self.C: self.C0, self.S: self.S0}

            if self.verbose:
                print('Optimizer created....')
                print('Minimizing the loss....')

            # Minimize the loss
            if self.optimizer == 'adam':
                self._minimize_with_adam(sess, optimizer, feed_dict)
            elif self.optimizer == 'l-bfgs':
                self._minimize_with_lbfgs(sess, optimizer, feed_dict)

            # Creating the generated image
            final_image = sess.run(self.X)

            return self.net.unpreprocess_input(final_image)

    def _build_graph(self):

        # Placeholders for content image and style_images
        self.C = tf.placeholder(
            dtype=tf.float32, shape=self.C0.shape, name='content_image')
        self.S = tf.placeholder(
            dtype=tf.float32, shape=self.S0.shape, name='style_images')

        # Trainable variable for generated_image
        self.X = tf.Variable(self.X0, name='generated_image')

        # Feed forward to compute generated layers at each layer in model
        generated_layers = self.net.forward(self.X, scope='generated')

        # Losses
        self.L_content = self._get_content_loss(generated_layers)
        self.L_style = self._get_style_loss(generated_layers)
        self.L_tv = self.tv * \
            tf.image.total_variation(self.X, name='t_variation_loss')[0]

        self.L_total = self.L_content + self.L_style + self.L_tv

    def _get_content_loss(self, generated_layers):

        # Feed forward to compute content layers at each layer in model
        content_layers = self.net.forward(self.C, scope='content')

        # Calculate loss for each content layer
        total_loss = 0.0

        # Iterate over all the content layers required
        for weight, layer in zip(self.content_layer_weights, self.content_layers):
            _, h, w, c = content_layers[layer].shape
            N, M = h.value * w.value, c.value

            c_loss = weight * \
                tf.reduce_sum(
                    tf.pow(content_layers[layer] - generated_layers[layer], 2))

            # Check for the content-loss-type
            if self.content_losstype == 1:
                c_loss *= 0.5
            elif self.content_losstype == 2:
                c_loss *= 1. / float(N * M)
            elif self.content_losstype == 3:
                c_loss *= 1. / (2. * N**0.5 * M**0.5)

            total_loss += c_loss

        total_loss /= len(self.content_layers)

        return self.alpha * total_loss

    def _get_style_loss(self, generated_layers):

        # Function to get the style loss for each style image
        def _get_loss(style_image):

            # Feed-forward to get the style layers at each layer in the model
            style_layers = self.net.forward(style_image, scope='style')
            style_loss = 0.0

            # Iterator over all the style layers required
            for weight, layer in zip(self.style_layer_weights, self.style_layers):
                _, h, w, c = style_layers[layer].shape
                N, M = h.value * w.value, c.value

                # Get gram matrix for each layer
                style_gram_mat = self._get_gram_matrix(style_layers[layer])
                gen_gram_mat = self._get_gram_matrix(generated_layers[layer])

                s_loss = weight * \
                    tf.reduce_sum(tf.pow(style_gram_mat - gen_gram_mat, 2))
                s_loss = 1. / (4. * N**2 * M**2) * s_loss
                style_loss += s_loss

            style_loss /= len(self.style_layers)

            return style_loss

        # iterate over all the style images and calculate loss
        total_loss = tf.map_fn(_get_loss, self.S, dtype=tf.float32)

        # Noramlize over all the style images
        total_loss = tf.reduce_mean(total_loss)

        return self.beta * total_loss

    def _get_optimizer(self):

        # Adam optimizer
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(self.lr)

        # L_BFGS optimizer
        elif self.optimizer == 'l-bfgs':
            return tf.contrib.opt.ScipyOptimizerInterface(
                self.L_total,
                method='L-BFGS-B',
                options={'maxiter': self.num_iter}
            )

    def _minimize_with_adam(self, sess, optimizer, feed_dict):

        # Initialize the optimizer
        train_op = optimizer.minimize(self.L_total)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Calculate and print the loss
        for it in range(1, self.num_iter + 1):
            content_loss, style_loss, tv_loss, total_loss, _ = sess.run(
                [self.L_content, self.L_style, self.L_tv, self.L_total, train_op],
                feed_dict=feed_dict
            )

            if it % self.print_iter == 0:
                print(f'At iteration: {it}  Content Loss: {content_loss}  Style Loss: {style_loss}  Tv Loss: {tv_loss}  Total Loss: {total_loss}')

    def _minimize_with_lbfgs(self, sess, optimizer, feed_dict):
        global _iter
        _iter = 0

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Callback fn to print the loss
        def _callback(c_loss, s_loss, tv_loss, t_loss):
            global _iter
            _iter += 1
            if _iter % self.print_iter == 0:
                print(f'At iteration: {_iter}  Content Loss: {c_loss}  Style Loss: {s_loss}  TV Loss: {tv_loss}  Total Loss: {t_loss}')

        # Initialize the optimizer
        optimizer.minimize(
            sess,
            feed_dict=feed_dict,
            fetches=[self.L_content, self.L_style, self.L_tv, self.L_total],
            loss_callback=_callback
        )

    def _get_gram_matrix(self, image):
        # get the number of channels in image
        num_c = int(image.shape[3].value)

        # Flatten the image to (H*W, C)
        image = tf.reshape(image, shape=[-1, num_c])

        return tf.matmul(tf.transpose(image), image)
