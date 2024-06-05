# Image Database Manager

## Descripción

Este proyecto proporciona una herramienta para gestionar una base de datos de imágenes y sus descriptores utilizando algoritmos de características SIFT y ORB. Incluye funcionalidades para insertar, eliminar, modificar y visualizar imágenes, así como para reconocer imágenes a partir de sus descriptores.

## Requisitos

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pillow (PIL)
- TensorFlow (si se requiere)

## Instalación

1. Clona el repositorio o descarga el código fuente.
2. Instala las dependencias utilizando `pip`:

    ```bash
    pip install numpy opencv-python matplotlib pillow tensorflow
    ```

## Uso

### Clase `dbEntry`

Esta clase representa una entrada en la base de datos, que incluye:
- `row_id`: Identificador único de la entrada.
- `img_array`: Array de la imagen.
- `sift_descriptors`: Descriptores SIFT de la imagen.
- `orb_descriptors`: Descriptores ORB de la imagen.
- `address`: Dirección (nombre del archivo) de la imagen.

### Clase `dbManager`

Esta clase maneja las operaciones en la base de datos de imágenes.

#### Métodos principales

- `load_db()`: Carga la base de datos desde el archivo especificado.
- `save_db()`: Guarda la base de datos en el archivo especificado.
- `insert(image_paths, siftParam, orbParam)`: Inserta nuevas imágenes en la base de datos.
- `delete(entry_id)`: Elimina una entrada de la base de datos por su ID.
- `modify(entry_id, new_image_path)`: Modifica una entrada existente en la base de datos.
- `find_entry_by_id(entry_id)`: Encuentra una entrada por su ID.
- `display_db()`: Muestra las entradas de la base de datos.
- `show_image(entry_id)`: Muestra una imagen específica por su ID.
- `show_all_images(images_per_row=5, thumbnail_size=(64, 64))`: Muestra todas las imágenes en la base de datos como miniaturas.
- `recognize_image_sift(imagen, numPuntos)`: Reconoce una imagen utilizando descriptores SIFT.
- `recognize_image_orb(imagen, numPuntos, umbral)`: Reconoce una imagen utilizando descriptores ORB.

### Ejemplo de Uso

```python
import pathlib
import cv2
import matplotlib.pyplot as plt
from db_manager import dbManager

# Inicializa el gestor de la base de datos
db_manager = dbManager()

# Directorio de imágenes
data_dir = pathlib.Path('./images/')
paths = list(data_dir.glob('*.jpeg'))
image_paths = [str(path) for path in paths]

# Inserta las imágenes en la base de datos
db_manager.insert(image_paths, 100, 100)

# Muestra todas las imágenes en la base de datos
db_manager.show_all_images()

# Reconoce una imagen
imagenOriginal = cv2.imread('./test/arcorect.jpg')
imagenOriginal = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2RGB)
best_match, num_matches = db_manager.recognize_image_sift(imagenOriginal, 500)
imagenMejorParecido = best_match.img_array
titulo = best_match.address

# Muestra la imagen reconocida
plt.figure()
plt.imshow(imagenMejorParecido)
plt.title(titulo)
plt.axis('off')
plt.show()
