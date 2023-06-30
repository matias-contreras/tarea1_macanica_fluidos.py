import numpy as np
import matplotlib.pyplot as plt

# constantes:
beta1 = 45 # angulo grados
beta11 = 45 * np.pi / 180 # angulo radianes
beta2 = 60  # angulo en grados
u = 70  # km/MA
sen = np.sin(beta11)
cos = np.cos(beta11)
tan = np.tan(beta11)
sencos = sen * cos

# constantes para la esquina del arco
a1 = 0
c1 = u * sen / beta1
d1 = 1/sen**2 * (c1 * (sencos - beta1) + u * sen)
b1 = -c1


# constantes para la esquina oceanica
a2 = 0
c2 = (u * sen) / (beta1 - sen)
d2 = 1/sen**2 * (c1 * (sencos - beta1) + u * sen)
b2 = -c1 - u


c1_libro = -np.pi * u * np.sqrt(2) / 2 / (2 - np.pi**2 / 4)  # para la esquina del arco
d1_libro = -u * np.sqrt(2) * (2 - np.pi / 2) / (2 - np.pi**2 / 4)  # esquina del arco
b1_libro = -c1_libro  # esquina de arco


w = 3 * np.pi / 2
p = w**2 - 2
q = 1 + w
r = np.sqrt(2)/q
s = np.sqrt(2) * (2 + w)
t = 2 * (1 + w)


c2_libro = u/p * (2 - r * (w + w**2))  # para la esquina oceanica
d2_libro = u/p * (s - t)  # para la esquina oceanica
b2_libro = -c2_libro - u  # para la esquina oceanica


def corriente(x, z, a, b, c, d):
    """

    :param x: coordenada eje x
    :param z: coordenada eje z (negativo hacia abajo)
    :param a: constante A_i
    :param b: constante B_i
    :param c: constante C_i
    :param d: constante D_i
    :return: funcion corriente dada en el enuncaiado
    """
    p = a * x
    q = b * z
    r = c * x
    s = d * z
    theta = np.arctan2(z, x)*180/np.pi

    return p + q + (r + s) * theta



N = 30  #Numero de elementos en la grilla (i.e. grilla de NxN)
x = np.linspace(-100, 100, N)
z = np.linspace(0, 100, N)
X, Z = np.meshgrid(x, -z)  #Grilla



arctan = np.arctan2(Z, X) * 180/np.pi
psi1 = corriente(X, Z, 0, b1, c1, d1)
psi2 = corriente(X, Z, 0, b2, c2, d2)

psi_final = np.zeros([N, N])
for i in range(N):
    for k in range(N):
        if Z[i, k] >= X[i, k] * -np.tan(beta11):
            psi_final[i, k] = psi1[i, k]
        if Z[i, k] <= X[i, k] * -np.tan(beta11):
            psi_final[i, k] = np.nan




plt.contourf(X, Z, arctan, levels=9, cmap='Spectral') #Graficar lineas de contorno
plt.colorbar()
plt.show()

plt.contourf(X, Z, psi_final, levels=20, cmap='Spectral') #Graficar lineas de contorno
plt.colorbar()
plt.show()




