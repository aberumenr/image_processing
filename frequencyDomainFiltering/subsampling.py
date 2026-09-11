import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread("Fine stripes.jpg")
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# submuestreo :p

# una si, una no COLUMNAS
sub_columnas = imagen[:, ::2]

# una si, una no FILAS
sub_filas = imagen[::2, :]

# una si, una no FILAS y COLUMNAS
sub_filas_columnas = imagen[::2, ::2]

fig, ejes = plt.subplots(2, 2, figsize=(12, 10))

ejes[0, 0].imshow(imagen)
ejes[0, 0].set_title(
    f"original pic\n{imagen.shape[1]} × {imagen.shape[0]} pixels"
)
'''
ejes[0, 1].imshow(sub_columnas)
ejes[0, 1].set_title(
    f"Submuestreo de columnas\n{sub_columnas.shape[1]} × "
    f"{sub_columnas.shape[0]} pixels"
)

ejes[1, 0].imshow(sub_filas)
ejes[1, 0].set_title(
    f"Submuestreo de filas\n{sub_filas.shape[1]} × "
    f"{sub_filas.shape[0]} pixels"
)'''

ejes[1, 1].imshow(sub_filas_columnas)
ejes[1, 1].set_title(
    f"subsampling\n"
    f"{sub_filas_columnas.shape[1]} × "
    f"{sub_filas_columnas.shape[0]} pixels"
)

for eje in ejes.flat:
    eje.axis("off")

plt.tight_layout()
plt.show()