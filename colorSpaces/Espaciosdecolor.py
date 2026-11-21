
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. GENERAR IMAGEN RGB
# ============================================================
alto = 300
ancho = 300

imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)
for y in range(alto):
    for x in range(ancho):
        R = int(255 * x / (ancho - 1))
        G = int(255 * y / (alto - 1))
        B = int(255 * (1 - x / (ancho - 1)))
        imagen[y, x] = [R, G, B]
RGB = imagen.astype(float) / 255.0
# ============================================================
# 2. SEPARACION DE CANALES RGB
# ============================================================
R = RGB[:, :, 0]
G = RGB[:, :, 1]
B = RGB[:, :, 2]
print("\n========== IMAGEN ORIGINAL ==========")
print("Dimensiones:", imagen.shape)
print("Canal R:\n", R)
print("Canal G:\n", G)
print("Canal B:\n", B)
# ============================================================
# 3. CONVERSION RGB A HSI
# ============================================================
I = (R + G + B) / 3.0
min_RGB = np.minimum(np.minimum(R, G), B)
S = np.zeros_like(I)
mask = I > 0
S[mask] = 1 - min_RGB[mask] / I[mask]
numerador = 0.5 * ((R - G) + (R - B))
denominador = np.sqrt((R-G)**2 + (R-B)*(G-B))
epsilon = 1e-10
theta = np.arccos(np.clip(numerador / (denominador + epsilon),-1, 1))
H = np.where(B <= G, theta, 2*np.pi-theta)
H = np.degrees(H)
H[S == 0] = 0
HSI = np.stack((H, S, I), axis=2)
# ============================================================
# 4. CONVERSION RGB A CIELAB
# ============================================================
# RGB sRGB a RGB lineal
RGB_lineal = np.where(RGB <= 0.04045,RGB / 12.92,((RGB + 0.055) / 1.055)**2.4)
Rl = RGB_lineal[:, :, 0]
Gl = RGB_lineal[:, :, 1]
Bl = RGB_lineal[:, :, 2]
# RGB lineal a XYZ D65
X = 0.4124564*Rl + 0.3575761*Gl + 0.1804375*Bl
Y = 0.2126729*Rl + 0.7151522*Gl + 0.0721750*Bl
Z = 0.0193339*Rl + 0.1191920*Gl + 0.9503041*Bl
# Blanco de referencia D65
Xn = 0.95047
Yn = 1.00000
Zn = 1.08883
Xr = X / Xn
Yr = Y / Yn
Zr = Z / Zn
delta = 6/29
def funcion_lab(t):
    return np.where(t > delta**3,np.cbrt(t),t/(3*delta**2) + 4/29)
fx = funcion_lab(Xr)
fy = funcion_lab(Yr)
fz = funcion_lab(Zr)
L = 116*fy - 16
a = 500*(fx-fy)
b = 200*(fy-fz)
CIELAB = np.stack((L, a, b), axis=2)
# ============================================================
# 5. CONVERSION RGB A CYM
# ============================================================
C = 1-R
Yc = 1-G
M = 1-B
CYM = np.stack((C, Yc, M), axis=2)
# ============================================================
# 6. MOSTRAR IMAGEN Y CANALES
# ============================================================
fig, ax = plt.subplots(3, 4, figsize=(15, 11))
imagenes = [
    (RGB, "Imagen RGB", None),
    (R, "Canal R", "gray"),
    (G, "Canal G", "gray"),
    (B, "Canal B", "gray"),

    (H, "Matiz H", "hsv"),
    (S, "Saturacion S", "gray"),
    (I, "Intensidad I", "gray"),
    (L, "CIELAB L*", "gray"),

    (a, "CIELAB a*", "gray"),
    (b, "CIELAB b*", "gray"),
    (C, "CYM Cyan", "gray"),
    (Yc, "CYM Yellow", "gray")
]

for eje, (datos, titulo, mapa) in zip(ax.flat, imagenes):
    if mapa is None:
        eje.imshow(datos)
    elif titulo == "Matiz H":
        eje.imshow(datos, cmap=mapa, vmin=0, vmax=360)
    elif titulo == "CIELAB L*":
        eje.imshow(datos, cmap=mapa, vmin=0, vmax=100)
    elif titulo == "CIELAB a*" or titulo == "CIELAB b*":
        eje.imshow(datos, cmap=mapa, vmin=-128, vmax=127)
    else:
        eje.imshow(datos, cmap=mapa, vmin=0, vmax=1)
    eje.set_title(titulo)
    eje.axis("off")
plt.tight_layout()
plt.show()
# Mostrar canal Magenta
plt.figure(figsize=(5, 5))
plt.imshow(M, cmap="gray", vmin=0, vmax=1)
plt.title("Canal CYM - Magenta")
plt.axis("off")
plt.show()
# ============================================================
# 7. CONVERTIR IMAGENES EN COORDENADAS
# ============================================================
pixeles_RGB = RGB.reshape(-1, 3)
pixeles_HSI = HSI.reshape(-1, 3)
pixeles_LAB = CIELAB.reshape(-1, 3)
pixeles_CYM = CYM.reshape(-1, 3)
colores = np.clip(pixeles_RGB, 0, 1)
total = len(pixeles_RGB)
# ============================================================
# 8. IMPRIMIR COORDENADAS DE TODOS LOS PIXELES
# ============================================================
print("\n========================================")
print("        COORDENADAS DE PIXELES")
print("========================================")
for i in range(total):
    print(
        f"\nPIXEL {i}:"
        f"\nRGB: R={pixeles_RGB[i,0]:.4f}, "
        f"G={pixeles_RGB[i,1]:.4f}, "
        f"B={pixeles_RGB[i,2]:.4f}"

        f"\nHSI: H={pixeles_HSI[i,0]:.2f}, "
        f"S={pixeles_HSI[i,1]:.4f}, "
        f"I={pixeles_HSI[i,2]:.4f}"

        f"\nLAB: L={pixeles_LAB[i,0]:.2f}, "
        f"a={pixeles_LAB[i,1]:.2f}, "
        f"b={pixeles_LAB[i,2]:.2f}"

        f"\nCYM: C={pixeles_CYM[i,0]:.4f}, "
        f"Y={pixeles_CYM[i,1]:.4f}, "
        f"M={pixeles_CYM[i,2]:.4f}"
    )
# ============================================================
# 10. FUNCION PARA GRAFICAR ESPACIOS DE COLOR 3D
# ============================================================

def graficar_espacio(datos, colores, titulo,
                     eje_x, eje_y, eje_z):

    max_puntos = 10000

    if len(datos) > max_puntos:
        indices = np.linspace(0, len(datos)-1,max_puntos,dtype=int)
    else:
        indices = np.arange(len(datos))

    puntos = datos[indices]
    colores_puntos = colores[indices]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        puntos[:, 0],
        puntos[:, 1],
        puntos[:, 2],
        c=colores_puntos,
        s=8,
        alpha=0.8,
        depthshade=False)

    ax.set_title(titulo, fontsize=15)
    ax.set_xlabel(eje_x)
    ax.set_ylabel(eje_y)
    ax.set_zlabel(eje_z)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
# ============================================================
# 11. GRAFICA ESPACIO RGB
# ============================================================
graficar_espacio(
    pixeles_RGB,
    colores,
    "ESPACIO RGB",
    "Rojo (R)",
    "Verde (G)",
    "Azul (B)")
# ============================================================
# 12. GRAFICA ESPACIO HSI
# ============================================================
graficar_espacio(
    pixeles_HSI,
    colores,
    "ESPACIO HSI",
    "Matiz H (grados)",
    "Saturacion S",
    "Intensidad I")
# ============================================================
# 13. GRAFICA ESPACIO CIELAB
# ============================================================
graficar_espacio(
    pixeles_LAB,
    colores,
    "ESPACIO CIELAB",
    "Luminosidad L*",
    "Eje a*",
    "Eje b*")
# ============================================================
# 14. GRAFICA ESPACIO CYM
# ============================================================
graficar_espacio(
    pixeles_CYM,
    colores,
    "ESPACIO CYM",
    "Cian (C)",
    "Amarillo (Y)",
    "Magenta (M)")


# Canal Rojo
plt.figure()
plt.imshow(R, cmap="Reds", vmin=0, vmax=1)
plt.title("Canal Rojo")
plt.colorbar()
plt.show()

# Canal Verde
plt.figure()
plt.imshow(G, cmap="Greens", vmin=0, vmax=1)
plt.title("Canal Verde")
plt.colorbar()
plt.show()

# Canal Azul
plt.figure()
plt.imshow(B, cmap="Blues", vmin=0, vmax=1)
plt.title("Canal Azul")
plt.colorbar()
plt.show()


# Matiz
plt.figure()
plt.imshow(H, cmap="hsv", vmin=0, vmax=360)
plt.title("Matiz H")
plt.colorbar()
plt.show()

# Saturacion
plt.figure()
plt.imshow(S, cmap="gray", vmin=0, vmax=1)
plt.title("Saturacion S")
plt.colorbar()
plt.show()

# Intensidad
plt.figure()
plt.imshow(I, cmap="gray", vmin=0, vmax=1)
plt.title("Intensidad I")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(L, cmap="gray", vmin=0, vmax=100)
plt.title("CIELAB - L*")
plt.show()

plt.figure()
plt.imshow(a, cmap="RdYlGn", vmin=-128, vmax=127)
plt.title("CIELAB - a*")
plt.show()

plt.figure()
plt.imshow(b, cmap="RdYlBu_r", vmin=-128, vmax=127)
plt.title("CIELAB - b*")
plt.show()


plt.figure()
plt.imshow(C, cmap="winter", vmin=0, vmax=1)
plt.title("Canal Cyan")
plt.show()

plt.figure()
plt.imshow(Yc, cmap="Wistia", vmin=0, vmax=1)
plt.title("Canal Yellow")
plt.show()

plt.figure()
plt.imshow(M, cmap="spring", vmin=0, vmax=1)
plt.title("Canal Magenta")
plt.show()
