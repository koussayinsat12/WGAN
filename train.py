from model import *
from config import *
from utils import *
import tensorflow_datasets as tfds
import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(f"Number of examples: {len(train_images)}")
print(f"Shape of the images in the dataset: {train_images.shape[1:]}")
train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype("float32")
train_images = (train_images - 127.5) / 127.5


generator=get_generator_model()
discriminator=get_discriminator_model()
generator.summary()
discriminator.summary()

ganMonitor=GANMonitor()
wgan=WGAN(generator=generator,discriminator=discriminator,latent_dim=dim_noise)
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    d_loss_fn=discriminator_loss,
    g_loss_fn=generator_loss
)
wgan.fit(train_images,epochs=epochs,callbacks=[ganMonitor]
)
