# main.py
import os
from utils.data_loader import DataLoader
from utils.data_augmentation import DataAugmentation

def main():
    base_dir = os.getcwd()  # Utilise le dossier courant comme base
    data_loader = DataLoader(base_dir)
    data_augmentation = DataAugmentation()

    data = data_loader.load_data()

    for image, annotations in data:
        augmented_image = data_augmentation.augment_image(image)
        # Ici, vous pouvez sauvegarder augmented_image et utiliser annotations comme nécessaire
        # Par exemple, sauvegarder l'image augmentée dans un dossier spécifique

if __name__ == "__main__":
    main()
