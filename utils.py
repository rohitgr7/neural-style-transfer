import numpy as np
import PIL
import matplotlib.pyplot as plt


def get_content_image(content_image, noise_ratio, seed=None):

    if seed:
        np.random.seed(seed)

    image = np.random.normal(size=content_image.shape, scale=1e-3)
    image = noise_ratio * image + (1 - noise_ratio) * content_image
    return np.float32(image)


def load_image(file_path, shape=None, max_size=None):

    image = PIL.Image.open(file_path)

    if max_size:
        factor = max_size / np.amax(image.size, axis=-1)
        new_size = (np.array(image.size) * factor).astype(np.int32)
        image = image.resize(new_size, PIL.Image.LANCZOS)

    if shape:
        image = image.resize(shape, PIL.Image.LANCZOS)

    image = np.float32(image)
    return image


def save_image(file_path, image):

    image = np.clip(image, 0.0, 255.0)

    if image.shape[2] == 1:
        image = image.reshape(image.shape[:2])

    image = image.astype(np.uint8)
    image = PIL.Image.fromarray(image)

    image.save(file_path)


def plot_images(content_image, style_image, generated_image):

    figure, axes = plt.subplots(1, 3, figsize=(10, 10))

    axes[0].set_title('Content Image')
    axes[0].imshow(content_image / 255., interpolation='sinc')

    axes[1].set_title('Style Image')
    axes[1].imshow(style_image / 255., interpolation='sinc')

    axes[2].set_title('Generated Image')
    axes[2].imshow(generated_image / 255., interpolation='sinc')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
