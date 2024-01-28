from keras.optimizers import RMSprop
IMG_SHAPE=(28,28,1)
dim_noise=128
epochs=100
batch_size=512
buffer_size=1000

generator_optimizer =RMSprop(learning_rate=0.00005)
discriminator_optimizer =RMSprop(learning_rate=0.00005)