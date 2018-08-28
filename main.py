import os

from parse import parser
from vgg import VGG
from stylize import Stylize
from utils import *


def main():

    args = parser()

    if args is None:
        exit()

    if args.verbose:
        print('Arguments parsed....')

    # Model instance
    vgg = VGG(args.model_path, args.pool_type, args.lalpha)

    if args.verbose:
        print('Model created....')

    # Content and Style Images
    content_image = load_image(os.path.join(args.content_path, args.content_image), max_size=args.max_size)
    style_images = [
        load_image(os.path.join(args.style_path, image),
                   shape=(content_image.shape[1], content_image.shape[0]))
        for image in args.style_images
    ]

    if args.verbose:
        print('Content and style images loaded....')

    if args.initial_type == 'content':
        init_gen_image = content_image
    elif args.initial_type == 'style':
        init_gen_image = style_images[0]
    elif args.initial_type == 'random':
        init_gen_image = get_content_image(content_image, args.noise_ratio, args.seed)

    if args.verbose:
        print('Generated image initialized....')

    # Stylize instance
    stylize = Stylize(vgg, content_image, style_images, init_gen_image, args)

    if args.verbose:
        print('Style-model created....')
        print('Generating image....')

    # Transfer style
    gen_image = stylize.transfer_style()

    if args.verbose:
        print('Image generated....')

    # Saving the image to destination path
    save_image(args.out_filepath, gen_image)

    if args.verbose:
        print('Generated image saved....')
        print('Completed!!!! :)')


if __name__ == '__main__':
    main()
