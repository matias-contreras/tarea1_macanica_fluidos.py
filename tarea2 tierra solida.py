import numpy as np
import matplotlib.pyplot as plt

# constantes:
beta1 = -45 * np.pi / 180  # angulo grados
beta11 = -45 * np.pi / 180  # angulo radianes
u = 70  # km/MA
sen = np.sin(beta11)
cos = np.cos(beta11)
tan = np.tan(beta11)
sencos = sen * cos

# constantes para la esquina del arco
a1 = 0
d1 = u * (beta1 * cos - sen) / (sen**2 - beta1**2)
c1 = (u * cos + d1 * (beta1 + sencos)) / sen**2
b1 = -c1


# constantes para la esquina oceanica
a2 = 0
d2 = u * ((cos - 1) * (beta1 - sencos) - sen**3) / (sen**2 - beta1**2)
c2 = (u * (cos - 1) + d2 * (beta1 + sencos)) / sen**2
b2 = -c2 - u


c1_libro = -np.pi * u * np.sqrt(2) / 2 / (2 - np.pi**2 / 4)  # para la esquina del arco
d1_libro = -u * np.sqrt(2) * (2 - np.pi/2) / (2 - np.pi**2/4)  # esquina del arco
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
    theta = np.arctan2(z, x)

    return p + q + (r + s) * theta



N = 30  #Numero de elementos en la grilla (i.e. grilla de NxN)
x = np.linspace(-100, 100, N)
z = np.linspace(0, 100, N)
X, Z = np.meshgrid(x, -z)  #Grilla

x2 = np.linspace(-100, 100, N)
z2 = np.linspace(0, 100, N)
X2, Z2 = np.meshgrid(-x2, -z2)  #Grilla



arctan = np.arctan2(Z, X) * 180/np.pi
psi1 = corriente(X, Z, 0, b1, c1, d1)
psi2 = corriente(X2, Z2, 0, b2_libro, c2_libro, d2_libro)


psi_final = np.zeros([N, N])
for i in range(N):
    for k in range(N):
        if Z[i, k] >= X[i, k] * np.tan(beta11):
            psi_final[i, k] = psi1[i, k]
        if Z[i, k] <= X[i, k] * np.tan(beta11):
            psi_final[i, k] = psi2[i, k]



plt.figure(figsize=(13, 5))
plt.contourf(X, Z, psi_final, levels=20, cmap='Spectral') #Graficar lineas de contorno
plt.axline((0, 0), (100, -100), linewidth = 8 , color='darkgoldenrod')
plt.title('dominio esquina del arco')
plt.colorbar()
plt.show()

plt.figure(figsize=(13, 5))
plt.contourf(X, Z, arctan, levels=20, cmap='Spectral') #Graficar lineas de contorno
plt.colorbar()
plt.show()

#plt.figure(figsize=(13, 5))
plt.contourf(X2, Z2, psi2, levels=20, cmap='Spectral') #Graficar lineas de contorno
plt.title('dominio esquina oceanica')
plt.colorbar()
plt.show()




