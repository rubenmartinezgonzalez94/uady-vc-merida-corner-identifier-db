import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import pathlib
from PIL import Image


class dbEntry:
    def __init__(self, row_id, img_array, sift_descriptors, orb_descriptors, address):
        self.row_id = row_id
        self.img_array = img_array
        self.sift_descriptors = sift_descriptors
        self.orb_descriptors = orb_descriptors
        self.address = address


class dbManager:
    def __init__(self, db_file='./corners_descriptors_db.npy'):
        self.db_file = db_file
        self.db = []
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_file):
            try:
                self.db = np.load(self.db_file).tolist()
            except (EOFError, ValueError, pickle.UnpicklingError):
                print(f"Failed to load {self.db_file}. Initializing an empty database.")
                self.db = []
        else:
            print(f"Database file not found. Initializing an empty database.")

    def save_db(self):
        np.save(self.db_file, self.db)

    def insert(self, image_paths, siftParam, orbParam):
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image at {image_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img)
            filename = os.path.basename(image_path)
            address = filename
            sift_descriptors = self.compute_sift_descriptors(img_array, siftParam)
            orb_descriptors = self.compute_orb_descriptors(img_array, orbParam)
            new_id = self.generate_id()
            entry = dbEntry(new_id, img_array, sift_descriptors, orb_descriptors, address)
            self.db.append(entry)
        self.save_db()

    def delete(self, entry_id):
        self.db = [entry for entry in self.db if entry.row_id != entry_id]
        self.save_db()

    def find_entry_by_id(self, entry_id):
        for entry in self.db:
            if entry.row_id == entry_id:
                return entry
        return None

    def find_entry_by_descriptors(self, query_descriptors):
        for entry in self.db:
            if entry.descriptors == query_descriptors:
                return entry
        return None

    def modify(self, entry_id, new_image_path):
        entry = self.find_entry_by_id(entry_id)
        if entry is None:
            print(f"Entry with ID {entry_id} not found")
        self.delete(entry_id)
        self.insert([new_image_path])

    def generate_id(self):
        if self.db:
            return self.db[-1].row_id + 1
        else:
            return 1

    def compute_sift_descriptors(self, image, numPoints):
        # This is a placeholder for the actual descriptor computation method
        # Replace with the appropriate method to compute descriptors for your images
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(numPoints)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return descriptors

    def compute_orb_descriptors(self, image, numPoints):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(numPoints)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return descriptors

    def display_db(self):
        for entry in self.db:
            print(f"ID: {entry.row_id}, Address: {entry.address}")

    def show_image(self, entry_id):
        entry = self.find_entry_by_id(entry_id)
        if entry is None:
            print(f"Entry with ID {entry_id} not found")
        plt.figure()
        plt.imshow(entry.img_array)
        plt.axis('off')
        plt.show()

    def show_all_images(self, images_per_row=5, thumbnail_size=(64, 64)):
        num_images = len(self.db)
        if num_images == 0:
            print("No hay imágenes en la base de datos.")
            return

        num_rows = (num_images + images_per_row - 1) // images_per_row

        # Create a figure for the collage
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row, num_rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for i, entry in enumerate(self.db):
            # Resize the image to the thumbnail size
            img = entry.img_array
            img_thumbnail = np.array(Image.fromarray(img).resize(thumbnail_size))

            # Show the image on the corresponding subplot
            axes[i].imshow(img_thumbnail)
            axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.show()

    def recognize_image(self, image, method, numPoints, umbral=None):
        best_match = None
        if method == 'sift':
            descriptors = self.compute_sift_descriptors(image, numPoints)
            best_match = self.recognize_image_sift(descriptors)
        elif method == 'orb':
            descriptors = self.compute_orb_descriptors(image, numPoints)
            best_match = self.recognize_image_orb(descriptors, umbral)
        else:
            print("Invalid method")
            return None

        return best_match

    def recognize_image_sift(self, descriptors):
        # Crea la instancia del comparador de fuerza bruta (BFMatcher)
        comparadorBF = cv2.BFMatcher_create()
        # Itera las entradas en base de datos para encontrar la mejor coincidencia
        maximoNumeroBuenasParejas = -1  # Inicializa el maximo
        mejorParecido = None  # Inicializa el mejor parecido
        for entry in self.db:
            # Obtiene los descriptores de entrada
            descriptoresEntrada = entry.sift_descriptors
            # Encuentra las parejas encontradas
            parejas = comparadorBF.knnMatch(descriptors, descriptoresEntrada, k=2)

            # Determina si es una buena pareja
            buenasParejas = []
            for m, n in parejas:
                if m.distance < 0.75 * n.distance:
                    buenasParejas.append([m])

            # Encuentra el mejor parecido segun la cantidad de parejas
            numeroBuenasParejas = len(buenasParejas)
            # print(numeroBuenasParejas, len(parejas))
            if maximoNumeroBuenasParejas < numeroBuenasParejas:
                maximoNumeroBuenasParejas = numeroBuenasParejas
                mejorParecido = entry
        return mejorParecido

    def recognize_image_orb(self, descriptors, umbral):
        # Crea la instancia del comparador de fuerza bruta (BFMatcher)
        comparadorBF = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        # Itera las entradas en base de datos para encontrar la mejor coincidencia
        maximoNumeroBuenasParejas = -1  # Inicializa el maximo
        mejorParecido = None  # Inicializa el mejor parecido
        for entry in self.db:
            # Obtiene los descriptores de entrada
            descriptoresEntrada = entry.orb_descriptors
            # Encuentra las parejas encontradas
            parejas = comparadorBF.match(descriptors, descriptoresEntrada)

            # Selecciona las mejores coincidencias
            buenasParejas = []
            for coincidencia in parejas:
                if coincidencia.distance <= umbral:
                    buenasParejas.append(coincidencia)

            # Encuentra el mejor parecido segun la cantidad de parejas
            numeroBuenasParejas = len(buenasParejas)
            # print(numeroBuenasParejas, len(parejas))
            if maximoNumeroBuenasParejas < numeroBuenasParejas:
                maximoNumeroBuenasParejas = numeroBuenasParejas
                mejorParecido = entry

        return mejorParecido

    # Example usage:


if __name__ == "__main__":
    db_manager = dbManager()

    data_dir = pathlib.Path('./images/')
    paths = list(data_dir.glob('*.jpeg'))
    image_paths = [str(path) for path in paths]
    # Insert images
    db_manager.insert(image_paths, 100, 100)

    # Delete an entry
    # db_manager.delete(1)

    # Modify an entry
    # db_manager.modify(2, './images/el_aguacate.jpg')

    # Display database entries
    #db_manager.display_db()

    # db_manager.show_image(1)

    db_manager.show_all_images()
