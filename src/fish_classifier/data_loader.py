import tensorflow as tf

def create_generators(data_dir, image_size=(224, 224), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen