from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataAugmentation:
    def __init__(self):
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def augment_image(self, image):
        # Convertir l'image PIL en array numpy pour l'augmentation
        image_array = np.array(image)
        
        # S'assurer que l'image a 4 dimensions (batch_size, height, width, channels)
        if image_array.ndim == 2:  # Image en niveaux de gris
            image_array = image_array[..., np.newaxis]
        
        # Keras s'attend à un batch dans la première dimension, donc nous ajoutons une dimension supplémentaire
        image_array = image_array[np.newaxis, ...]
        
        # Appliquer l'augmentation
        it = self.datagen.flow(image_array, batch_size=1)
        batch = next(it)
        
        # Retirer la dimension de batch et convertir de retour en image PIL
        augmented_image = Image.fromarray(batch[0].astype('uint8').squeeze())
        return augmented_image
