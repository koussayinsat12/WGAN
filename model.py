import os

os.environ['KERAS_BACKEND']="tensorflow"
import keras
import tensorflow as tf
from keras import layers
from config import *
from keras.models import Model

class WGAN(Model):
    def __init__(
        self,generator,discriminator,latent_dim,discriminator_extra_steps=3,gp_weight=10.0
    ):
        super().__init__()
        self.generator=generator
        self.discriminator=discriminator
        self.latent_dim=latent_dim
        self.discriminator=discriminator
        self.d_steps=discriminator_extra_steps
        self.gp_weight=gp_weight
    
    def compile(self,d_optimizer,g_optimizer,d_loss_fn,g_loss_fn):
        super().compile()
        self.d_optimizer=d_optimizer
        self.g_optimizer=g_optimizer
        self.d_loss_fn=d_loss_fn
        self.g_loss_fn=g_loss_fn
    
    @tf.function()
    def gradient_penality(self,batch_size,real_images,fake_images):
        
        alpha=tf.random.uniform([batch_size,1,1,1],0.0,1.0)
        diff=fake_images-real_images
        interpolated=real_images+alpha*diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred=self.discriminator(interpolated,training=True)
        grads=gp_tape.gradient(pred,[interpolated])[0]
        norm=tf.sqrt(tf.reduce_mean(tf.square(grads),axis=[1,2,3]))
        gp=tf.reduce_mean((norm-1.0)**2)
        return gp

    @tf.function()
    def train_step(self,real_images):
        if isinstance(real_images,tuple):
            real_images=real_images[0]
        batch_size=tf.shape(real_images)[0]

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penality(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)           
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        




