import rectifykk
import sys, pathlib
import cv2
import matplotlib.pyplot as plt

sys.path.append('uady-vc-merida-corner-identifier-db')
from corners_descriptors_db import dbManager

db_manager = dbManager()

data_dir = pathlib.Path('uady-vc-merida-corner-identifier-db/images/DB')
paths = list(data_dir.glob('*.jpeg'))
image_paths = [str(path) for path in paths]

#with open('readme.txt', 'w') as f:
#    for line in image_paths:
#        f.write(line)
#        f.write('\n')

# Insert images
db_manager.insert(image_paths, 3000, 3000)
#db_manager.show_all_images()

#image = cv2.imread('uady-vc-merida-corner-identifier-db/images/El_Globo_X_44_Y_61.jpeg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#plt.figure()
#plt.imshow(image)
#plt.show()

path1 = './test/esquina-el-cedro.jpg'
image2 = cv2.imread(path1)
plt.figure()
plt.imshow(image2)
plt.show()

best_match = rectifykk.matchSearch(db_manager, path1, 1, 'sift', 80)
#imagenMejorParecido = best_match.img_array
#titulo = best_match.address

## Mostrar la imagen con el título correspondiente
#plt.figure()
#plt.imshow(imagenMejorParecido)
#plt.title(titulo)  # Añadir el título de la imagen
#plt.axis('off')  # Ocultar los ejes
#plt.show()


#best_match = rectifykk.matchSearch(db_manager, path1, 1, 'orb', 15, 60)
if best_match != None: 
    imagenMejorParecido = best_match.img_array
    titulo = best_match.address

    # Mostrar la imagen con el título correspondiente
    plt.figure()
    plt.imshow(imagenMejorParecido)
    plt.title(titulo)  # Añadir el título de la imagen
    plt.axis('off')  # Ocultar los ejes
    plt.show()
else:
    print('No se encontró en la base de datos')


