# Hands-On-Image-Generation-with-TensorFlow
Hands-On Image Generation with TensorFlow, published by Packt
https://www.amazon.com/dp/1838826785

The code in this book uses tensorflow_gpu==2.2.0

1. Getting Started with Image Generation with Tensorflow
- PixelCNN

2. Variational Autoencoder (VAE)
- Autoencoder
- VAE

3. Genenrative Adversarial Network (GAN)
- DCGAN
- WGAN
- WGAN-GP

4. Image-to-Image Translation
- Conditional DCGAN
- pix2pix
- CycleGAN
- BicycleGAN

5. Style Transfer
- Neural Style Transfer
- Arbitrary Style Transfer with AdaIN

6. AI Painter
- iGAN
- GauGAN

7. High Fidelity Image Generation
- Progressive GAN
- Style GAN

8. Attention-based Generative Models
- Self-Attention GAN (SAGAN)
- BigGAN

9. Video Synthesis
- DeepFake

## Installation
#### Create Virtual Environment
python3 -m venv ./venv/imgentf2

#### Source Virtual environment
source ./venv/imgentf2/bin/activate

#### Install dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

#### Add virtual environment into jupyter notebook
python -m ipykernel install --user --name=imgentf2

#### Enable jupyter notebook extension
jupyter nbextension enable --py widgetsnbextension

