## $5 Tech Unlocked 2021!
[Buy and download this product for only $5 on PacktPub.com](https://www.packtpub.com/)
-----
*The $5 campaign         runs from __December 15th 2020__ to __January 13th 2021.__*


# Hands-On Image Generation with TensorFlow

<a href="https://www.packtpub.com/product/hands-on-image-generation-with-tensorflow/9781838826789?utm_source=github&utm_medium=repository&utm_campaign=9781838826789"><img src="https://static.packt-cdn.com/products/9781838826789/cover/smaller" alt="Hands-On Image Generation with TensorFlow" height="256px" align="right"></a>

This is the code repository for [Hands-On Image Generation with TensorFlow](https://www.packtpub.com/product/hands-on-image-generation-with-tensorflow/9781838826789?utm_source=github&utm_medium=repository&utm_campaign=9781838826789), published by Packt.

**A practical guide to generating images and videos using deep learning**

## What is this book about?
The emerging field of Generative Adversarial Networks (GANs) has made it possible to generate indistinguishable images from existing datasets. With this hands-on book, you’ll not only develop image generation skills but also gain a solid understanding of the underlying principles.

Starting with an introduction to the fundamentals of image generation using TensorFlow, this book covers Variational Autoencoders (VAEs) and GANs. You’ll discover how to build models for different applications as you get to grips with performing face swaps using deepfakes, neural style transfer, image-to-image translation, turning simple images into photorealistic images, and much more. You’ll also understand how and why to construct state-of-the-art deep neural networks using advanced techniques such as spectral normalization and self-attention layer before working with advanced models for face generation and editing. You'll also be introduced to photo restoration, text-to-image synthesis, video retargeting, and neural rendering. Throughout the book, you’ll learn to implement models from scratch in TensorFlow 2.x, including PixelCNN, VAE, DCGAN, WGAN, pix2pix, CycleGAN, StyleGAN, GauGAN, and BigGAN.

By the end of this book, you'll be well versed in TensorFlow and be able to implement image generative technologies confidently.


This book covers the following exciting features: 
* Train on face datasets and use them to explore latent spaces for editing new faces
* Get to grips with swapping faces with deepfakes
* Perform style transfer to convert a photo into a painting
* Build and train pix2pix, CycleGAN, and BicycleGAN for image-to-image translation
* Use iGAN to understand manifold interpolation and GauGAN to turn simple images into photorealistic images
* Become well versed in attention generative models such as SAGAN and BigGAN
* Generate high-resolution photos with Progressive GAN and StyleGAN

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/B08LVL4FPN) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
The code in this book uses tensorflow_gpu==2.2.0

The code will look like the following:
```
content_image = self.preprocess(content_image_input)
style_image = self.preprocess(style_image_input)
self.content_target = self.encoder(content_image)
self.style_target = self.encoder(style_image)
adain_output = AdaIN()([self.content_target[-1], self.style_target[-1]])
self.stylized_image = self.postprocess(self.decoder(adain_output))
self.stn = Model([content_image_input, style_image_input], self.stylized_image)

```
#### Create virtual environment
```python3 -m venv ./venv/imgentf2```

#### Source virtual environment
```source ./venv/imgentf2/bin/activate```

#### Install dependencies
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

#### Add virtual environment into Jupyter Notebook
```python -m ipykernel install --user --name=imgentf2```

#### Enable Jupyter Notebook extension
```jupyter nbextension enable --py widgetsnbextension```

**Following is what you need for this book:**
This book is a step-by-step guide to show you how to implement generative models in TensorFlow 2.x from scratch. You’ll get to grips with the image generative technology by covering autoencoders, style transfer, and GANs as well as fundamental and state-of-the-art models. You should have basic knowledge of deep learning training pipelines, such as training convolutional neural networks for image classification. This book will mainly use highlevel Keras APIs in TensorFlow 2, which is easy to learn. Should you need to refresh or learn TensorFlow 2, there are many free tutorials available online, such as the one on the official TensorFlow website, [https://www.tensorflow.org/tutorials/keras/classification](https://www.tensorflow.org/tutorials/keras/classification).

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
| 1 - 10   |   TensorFlow 2.2, GPU with minimum 4 GB memory                                       | Windows, Mac OS X, and Linux (Any) |

Training deep neural networks is computationally intensive. You can train the first few simple models using the CPU only. However, as we progress to more complex models and datasets in later chapters, the model training could take a few days before you start to see satisfactory results. To get the most out of this book, you should have access to the GPU to accelerate the model training time. There are also free cloud services, such as Google's Colab, that provide GPUs on which you can upload and run the code. 


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781838826789_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Generative Adversarial Networks Cookbook [[Packt]](https://www.packtpub.com/product/generative-adversarial-networks-cookbook/9781789139907) [[Amazon]](https://www.amazon.com/dp/1789139902)

* Python Image Processing Cookbook [[Packt]](https://www.packtpub.com/product/python-image-processing-cookbook/9781789537147) [[Amazon]](https://www.amazon.com/dp/1789537142)

## Get to Know the Author
**Soon Yau Cheong** is an AI consultant and the founder of Sooner.ai Ltd. With a history of being associated with industry giants such as NVIDIA and Qualcomm, he provides consultation in the various domains of AI, such as deep learning, computer vision, natural language processing, and big data analytics. He was awarded a full scholarship to study for his PhD at the University of Bristol while working as a teaching assistant. He is also a mentor for AI courses with Udacity.




