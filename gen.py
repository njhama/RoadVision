import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

def load_images(root_dir, target_size=(64, 64)):
    images = []
    for continent in os.listdir(root_dir):
        print(f"Checking in continent: {continent}")
        continent_path = os.path.join(root_dir, continent)

        if os.path.isdir(continent_path):
            for image_file in os.listdir(continent_path):
                image_path = os.path.join(continent_path, image_file)

                if os.path.isfile(image_path):
                    #print(f"Loading image: {image_path}")
                    image = load_img(image_path, target_size=target_size, color_mode='rgb')
                    image = img_to_array(image)
                    image /= 255.0
                    images.append(image)

    return np.array(images)




dataset_root = "./data"  
images = load_images(dataset_root)

train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
train_ds = tf.data.Dataset.from_tensor_slices(train_images).batch(64).shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(val_images).batch(64).prefetch(tf.data.AUTOTUNE)

def build_generator():
    model = Sequential()
    model.add(Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((8, 8, 256)))

    #16x16
    model.add(UpSampling2D())
    model.add(Conv2D(128, (5, 5), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    #32x32
    model.add(UpSampling2D())
    model.add(Conv2D(64, (5, 5), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    #  64x64
    model.add(UpSampling2D())
    model.add(Conv2D(64, (5, 5), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # 64x64x3
    model.add(Conv2D(3, (5, 5), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

generator = build_generator()
discriminator = build_discriminator()

class GMapsGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator 
        self.discriminator = discriminator 

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, 100))
        generated_images = self.generator(random_latent_vectors, training=True)
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as d_tape:
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(generated_images, training=True)
            d_loss_real = self.d_loss(real_labels, real_predictions)
            d_loss_fake = self.d_loss(fake_labels, fake_predictions)
            d_loss = (d_loss_real + d_loss_fake) / 2
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

    
        random_latent_vectors = tf.random.normal(shape=(batch_size, 100))
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as g_tape:
            fake_predictions = self.discriminator(self.generator(random_latent_vectors, training=True), training=False)
            g_loss = self.g_loss(misleading_labels, fake_predictions)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def call(self, inputs, training=False):
        generated_images = self.generator(inputs, training=training)
        return generated_images

# Compile  GAN
g_opt = Adam(learning_rate=0.0002, beta_1=0.5)
d_opt = Adam(learning_rate=0.0002, beta_1=0.5)
g_loss = BinaryCrossentropy(from_logits=False)
d_loss = BinaryCrossentropy(from_logits=False)

gmaps_gan = GMapsGAN(generator, discriminator)
gmaps_gan.compile(g_opt, d_opt, g_loss, d_loss)

def generate_and_save_images(model, epoch, test_input, folder='generated_images'):
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.figure(figsize=(6, 6))
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255)
        plt.axis('off')
        plt.savefig(os.path.join(folder, f'image_at_epoch_{epoch}_num_{i}.png'))
        plt.close()

class ModelMonitor(Callback):
    def __init__(self, num_img=4, latent_dim=100, save_folder='generated_images'):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)
        self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        generate_and_save_images(self.model.generator, epoch, self.random_latent_vectors, self.save_folder)


# Training  GAN
hist = gmaps_gan.fit(train_ds, epochs=10, callbacks=[ModelMonitor()])

# Gen images
generate_and_save_images(generator, 'final', tf.random.normal(shape=(4, 100)))
print('done')