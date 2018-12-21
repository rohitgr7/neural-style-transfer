import argparse


def parser():

    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        default='./models/imagenet-vgg-verydeep-19.mat',
        help='Preloaded VGG19 model weights path'
    )

    # Files arguments
    parser.add_argument(
        '--content_path',
        type=str,
        default='./images/content_images',
        help='Content image path'
    )

    parser.add_argument(
        '--style_path',
        type=str,
        default='./images/style_images',
        help='Style images path'
    )

    parser.add_argument(
        '--content_image',
        type=str,
        default='content2.jpg',
        help='Content image'
    )

    parser.add_argument(
        '--style_images',
        nargs='+',
        type=str,
        default=['style13.jpg'],
        help='Style images'
    )

    parser.add_argument(
        '--out_filepath',
        type=str,
        default='./images/generated_images/gen_image.jpg',
        help='Path where generated file will be stored'
    )

    # Parameers for loss function
    parser.add_argument(
        '--initial_type',
        choices=['content', 'style', 'random'],
        type=str,
        default='random',
        help='Initial type for the generated image'
    )

    parser.add_argument(
        '--content_layers',
        nargs='+',
        type=str,
        default=['conv4_2'],
        help='Content layers used for content loss'
    )

    parser.add_argument(
        '--style_layers',
        nargs='+',
        type=str,
        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        help='Style layers used for style loss'
    )

    parser.add_argument(
        '--content_losstype',
        choices=[1, 2, 3],
        type=int,
        default=3,
        help='Different types of content loss types'
    )

    parser.add_argument(
        '--content_layer_weights',
        nargs='+',
        type=float,
        default=[1.0],
        help='Content loss weight which are multiplied with its corresponding content layer'
    )

    parser.add_argument(
        '--style_layer_weights',
        nargs='+',
        type=float,
        default=[.2, .2, .2, .2, .2],
        help='Style loss weight which are multiplied with its corresponding style layer'
    )

    # Hyperparameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=5e0,
        help='Hyperparameter for content loss'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=1e2,
        help='Hyperparameter for style loss'
    )

    parser.add_argument(
        '--tv',
        type=float,
        default=1e-2,
        help='Hyperparameter for total variance loss'
    )

    parser.add_argument(
        '--optimizer',
        choices=['adam', 'l-bfgs'],
        type=str,
        default='l-bfgs',
        help='Optimizer to use for calculating gradients'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e2,
        help='Learning rate fo optimization'
    )

    parser.add_argument(
        '--num_iter',
        type=int,
        default=1000,
        help='Number of iterations for optimization'
    )

    parser.add_argument(
        '--noise_ratio',
        type=float,
        default=1.0,
        help='Noise ratio for random image with content image'
    )

    # Additional parameters
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for random generation'
    )

    parser.add_argument(
        '--pool_type',
        choices=['max', 'avg'],
        type=str,
        default='avg',
        help='Pooling type to use in VGG19 model'
    )

    parser.add_argument(
        '--lalpha',
        type=float,
        default=0.0,
        help='Alpha for l-relu'
    )

    parser.add_argument(
        '--max_size',
        type=int,
        default=300,
        help='Max of height or weight of the image to be generated'
    )

    parser.add_argument(
        '--print_iter',
        type=int,
        default=50,
        help='Print the loss after a certain number of iterations'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=1,
        help='Print the statements or not'
    )

    return parser.parse_args()
