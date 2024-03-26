import os
from pdf2image import convert_from_path
from shutil import move
from PIL import Image  # Importer la classe Image de Pillow

# Chemin du dossier contenant les PDFs
folder = 'C:\\Users\\ppbar\\Documents\\Projects\\ScanPID\\image training\\pdf'

# Créer le sous-dossier 'png' s'il n'existe pas
png_folder = os.path.join(folder, 'png')
os.makedirs(png_folder, exist_ok=True)

# Chemin vers le dossier où les PDFs traités seront déplacés
pdf_done_folder = os.path.join(folder, 'pdfInPngDone')

# Vérifier si le dossier pdfInPngDone existe, sinon le créer
os.makedirs(pdf_done_folder, exist_ok=True)

# Parcourir tous les fichiers PDF du dossier
for filename in os.listdir(folder):
    if filename.endswith('.pdf'):
        # Chemin complet du fichier PDF
        pdf_path = os.path.join(folder, filename)
        
        # Convertir le PDF en liste d'images (une par page)
        images = convert_from_path(pdf_path)
        
        # Enregistrer chaque page comme image PNG en noir et blanc
        for i, image in enumerate(images):
            # Convertir l'image en noir et blanc
            image_bw = image.convert('L')
            
            # Construire le nom de fichier PNG
            base_filename = filename[:-4]  # Enlever '.pdf'
            image_filename = f"{base_filename}-page_{i + 1}.png"
            image_path = os.path.join(png_folder, image_filename)
            
            # Sauvegarder l'image en noir et blanc
            image_bw.save(image_path, 'PNG')

        # Déplacer le PDF traité dans le dossier pdfInPngDone
        move(pdf_path, os.path.join(pdf_done_folder, filename))

        print(f"Fichier terminé et déplacé pour : {filename}")
