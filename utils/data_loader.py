# utils/data_loader.py
import json
import os
from PIL import Image

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.img_dir = os.path.join(base_dir, "image training", "valve", "img")
        self.json_dir = os.path.join(base_dir, "image training", "valve", "json")

    def load_data(self):
        data = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(self.img_dir, filename)
                json_path = os.path.join(self.json_dir, filename.replace('.png', '.json'))

                # Charger l'image
                image = Image.open(img_path)

                # Charger les annotations JSON
                with open(json_path, 'r') as f:
                    annotations = json.load(f)

                data.append((image, annotations))
        return data
