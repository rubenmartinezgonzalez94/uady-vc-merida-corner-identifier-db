import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import pathlib


class dbEntry:
    def __init__(self, row_id, img_array, descriptors, address):
        self.row_id = row_id
        self.img_array = img_array
        self.descriptors = descriptors
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

    def insert(self, image_paths):
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image at {image_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img)
            filename = os.path.basename(image_path)
            address = filename
            descriptors = self.compute_descriptors(img_array)
            #find_entry = self.find_entry_by_descriptors(descriptors)
            #if find_entry is not None:
            #    print(f"Entry already exists with ID {find_entry.row_id}")
            #    continue
            new_id = self.generate_id()
            entry = dbEntry(new_id, img_array, descriptors, address)
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
    def compute_descriptors(self, image):
        # This is a placeholder for the actual descriptor computation method
        # Replace with the appropriate method to compute descriptors for your images
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
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

    def show_all_images(self):
        for entry in self.db:
            plt.figure()
            plt.imshow(entry.img_array)
            plt.axis('off')
            plt.show()



# Example usage:
if __name__ == "__main__":
    db_manager = dbManager()

    data_dir = pathlib.Path('./images/')
    paths = list(data_dir.glob('*.jpg'))
    image_paths = [str(path) for path in paths]
    # Insert images
    db_manager.insert(image_paths)

    # Delete an entry
    # db_manager.delete(1)

    # Modify an entry
    # db_manager.modify(2, './images/el_aguacate.jpg')

    # Display database entries
    db_manager.display_db()

    # db_manager.show_image(1)

    db_manager.show_all_images()
