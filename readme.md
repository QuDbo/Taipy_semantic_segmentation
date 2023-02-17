# Simple Taipy application for a semantic segmentation demonstrator

This example show how to set up a simple Taipy application to present the result of an semantic segmentation algorithm.

## Taipy application details

To be completed

## Semantic segmentation algorithm details
<p align="center">
<a href="https://www.cityscapes-dataset.com/"><img src="./img_readme/cityscapes.png" alt="Cityscapes logo" width=800px/></a>
</p>  
The algorithm use in this example is trained on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/dataset-overview/) that represent urban street scene taken from a car point of view.

An example of data use for our model is presented below. The input data for the model are RGB images and the target data are semantic segmentation representation where each pixel of the image is labelled by a class id. The dataset contains 30 classes but we only use the 8 super classes in this case.

<p align="center">
<img src="./img_readme/example_cityscapes_data.png" alt="Example of data used" width=600px/>
</p>

The model provided is a trained Unet configuration with a VGG16 encoder as represented below. This model doesn't present the best performance for this kind of task but is quite simple and light which is better for a demonstrator.

<p align="center">
<img src="./img_readme/unet_with_vgg16.png" alt="Unet scheme with VGG6 encoder" width=700px/>
</p>