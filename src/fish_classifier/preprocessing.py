# Eğer preprocessing işlemlerine ihtiyacın varsa
# şimdilik normalizasyon zaten DataLoader'da yapılıyor
# ekstra augmentasyon eklenebilir

import tensorflow as tf

def augment_data(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image