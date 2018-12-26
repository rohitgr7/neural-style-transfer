# Neural Style Transfer

An implementation of Neural style transfer based on this [paper](http://arxiv.org/pdf/1508.06576v2.pdf) in TensorFlow.

This implementation uses [tensorflow v1.9](https://www.tensorflow.org/) using [python3.6](https://www.python.org/).

It is proposed to use [L-BFGS](l-bfgs) (which is what the original authors
used), but we can use [Adam][adam] with a little bit more hyperparameter tuning to get amazing results.

## Requirements

### Data Files
Pre-trained VGG19 network - use `bash get_data.sh` to download pretrained vgg19 model. Specify its location using the `--model_path` option.

### Dependencies

Use `pip install -r requirements.txt` to install all the required dependencies.

If you want to install the packages manually, here's the
list:

* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/install.html)
* [Pillow](https://pillow.readthedocs.io/en/5.3.x/installation.html)

## Guide
Use `python main.py --content_image <content file> --style_images <style files> --out_filepath <output file>` to stylize and image.

You can choose multiple style images to generate multi-styled output.

There are other options available which you can pass. You can run `python neural_style.py --help` to see a list of all options.

Use `--iter` to change the number of iterations (default 1000). Use `--max_size` (default 512) to change the output image size.

Running it for around 500-1000 iterations seems to produce amazing results. You can choose the content layers with `--content_layers` and style layers with `--style_layers` with layer weights for content layers and style layers using `--content_layer_weights` and `--style_layer_weights` respectively.

The model used is Pretrained VGG19 ([weights](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)).

## Results

<div align="center">
 <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content9.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/style_images/style14.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_10.jpg" width="512px">
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content12.jpg" height="223px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/style_images/style10.jpg" height="223px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_14.jpg" width="512px">
</div>

### More results

<div align="center">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content4.jpg" height="250px" width="300px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_3.jpg" height="250px" width="300px">

  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content1.jpg" height="250px" width="300px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_2.jpg" height="250px" width="300px">

  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content3.jpg" height="250px" width="300px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_11.jpg" height="250px" width="300px">

  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content14.jpg" height="250px" width="300px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_0.jpg" height="250px" width="300px">

  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content12.jpg" height="250px" width="300px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_5.jpg" height="250px" width="300px">

  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/content_images/content11.jpg" height="250px">
  <img src="https://raw.githubusercontent.com/rohitgr7/neural-style-transfer/master/images/generated_images/gen_18.jpg" height="250px">
</div>

## References

* Image Style Transfer Using Convolutional Neural Networks - [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* A Neural Algorithm of Artistic Style - [Paper](http://arxiv.org/pdf/1508.06576v2.pdf)
* Neural Artistic Style Transfer: A Comprehensive Look - [Link](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199)
* Creating Intricate Art with Neural Style Transfer - [Link](https://becominghuman.ai/creating-intricate-art-with-neural-style-transfer-e5fee5f89481)

