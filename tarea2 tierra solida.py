import numpy as np
import matplotlib.pyplot as plt

# constantes:
beta1 = -60 * np.pi / 180  # angulo grados
beta11 = -60 * np.pi / 180  # angulo radianes
u = 70/31536 * 10**-6  # m/s
eta = 10**21  # mks
sen = np.sin(beta11)
cos = np.cos(beta11)
tan = np.tan(beta11)
sencos = sen * cos

def ctes1(beta):
    d_1 = u * (beta * np.cos(beta) - np.sin(beta)) / (np.sin(beta)**2 - beta**2)
    c_1 = (u * np.cos(beta) + d_1 * (beta + np.sin(beta)*np.cos(beta))) / np.sin(beta)**2
    b_1 = -c_1
    a_1 = 0 * beta

    return np.array([a_1, b_1, c_1, d_1])

def ctes2(beta):
    sen = np.sin(beta)
    cos = np.cos(beta)
    sencos = sen * cos
    d_2 = u * (sen * cos ** 2 - (cos - 1) * (np.pi + beta - sencos)) / (np.pi + beta) ** 2
    c_2 = (u * sen + d_2 * sen ** 2) / (np.pi + beta - sencos)
    b_2 = d_2 * np.pi - c_2 - u
    a_2 = c_2 * np.pi
    return [a_2, b_2, c_2, d_2]


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


# constantes para la esquina del arco

a1 = ctes1(beta1)[0]
b1 = ctes1(beta1)[1]
c1 = ctes1(beta1)[2]
d1 = ctes1(beta1)[3]


# constantes para la esquina oceanica
a2 = ctes2(beta1)[0]
b2 = ctes2(beta1)[1]
c2 = ctes2(beta1)[2]
d2 = ctes2(beta1)[3]






def corriente(x, z, arctan, a, b, c, d):
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
    theta = arctan

    return p + q + (r + s) * theta

def velx(x, z, arctan, a, b, c, d):
    arctg = arctan
    dmult = d*(arctg + z * x/(x**2 + z**2))
    cmult = c * x**2/(x**2 + z**2)
    vx = -(b + dmult + cmult)
    return vx


def velz(x, z, arctan, a, b, c, d):
    arctg = arctan
    cmult = c * (arctg - x * z/(x**2 + z**2))
    dmult = d * z**2/(x**2 + z**2)
    vz = a + cmult - dmult
    return vz


def presion(radio, theta, c, d):
    x = radio * np.cos(theta)
    z = radio * np.sin(theta)
    p = -2 * eta * (c * x + d * z)/radio**2
    return p

def torque(presion, radio):
    t = presion * radio
    return t


N = 30  #Numero de elementos en la grilla (i.e. grilla de NxN)

x = np.linspace(-100*10**3, 100*10**3, N)
z = np.linspace(0, 100*10**3, N)
X, Z = np.meshgrid(x, -z)  #Grilla

x2 = np.linspace(-100*10**3, 100*10**3, N)
z2 = np.linspace(0, 100*10**3, N)
X2, Z2 = np.meshgrid(-x2, -z2)  #Grilla



#arctan = np.arctan2(Z, X) * 180/np.pi
#psi1 = corriente(X, Z, 0, b1, c1, d1)
#psi2 = corriente(X2, Z2, 0, b2, c2, d2)


psi_final = np.zeros([N, N])
velx_final = np.zeros([N, N])
velz_final = np.zeros([N, N])
arctans = np.zeros([N, N])
for i in range(N):
    for k in range(N):
        if Z[i, k] == 0:
            if X[i, k] == 0:
                arctans[i, k] = 0
            elif X[i, k] < 0:
                arctans[i, k] = -np.pi
            else:
                arctans[i, k] = 0
        else:
            arctans[i, k] = np.arctan2(Z[i, k], X[i, k])
        if Z[i, k] >= X[i, k] * np.tan(beta11):
            psi_final[i, k] = corriente(X[i, k], Z[i, k], arctans[i, k], a1, b1, c1, d1)
            velx_final[i, k] = velx(X[i, k], Z[i, k], arctans[i, k], a1, b1, c1, d1)
            velz_final[i, k] = velz(X[i, k], Z[i, k], arctans[i, k], a1, b1, c1, d1)

        if Z[i, k] <= X[i, k] * np.tan(beta11):
            psi_final[i, k] = corriente(X[i, k], Z[i, k], arctans[i, k], a2, b2, c2, d2)
            velx_final[i, k] = velx(X[i, k], Z[i, k], arctans[i, k], a2, b2, c2, d2)
            velz_final[i, k] = velz(X[i, k], Z[i, k], arctans[i, k], a2, b2, c2, d2)


presion_placa = np.abs(presion(np.sqrt(X[0][int(N/2):]**2 + (X[0][int(N/2):]*np.tan(beta1))**2),
                               beta1, c1, d1)) + np.abs(presion(np.sqrt(X[0][int(N/2):]**2 + X[0][int(N/2):]*np.tan(beta1)**2),
                                                                beta1, c2, d2))
torques = torque(presion_placa, np.sqrt(X[0][int(N/2):]**2 + (X[0][int(N/2):]*np.tan(beta1))**2))*10**-9



#plt.streamplot(X*10**-3,-Z*10**-3,velx_final,velz_final,color=psi_final, C=np.sqrt(velz_final**2+velx_final**2))
#plt.colorbar()
#plt.show()


plt.figure(figsize=(13, 5))
plt.contourf(X*10**-3, Z*10**-3, psi_final, levels=20, cmap='Spectral') #Graficar lineas de contorno
plt.plot(X[0]*10**-3, X[0]*10**-3*np.tan(beta1), linewidth=7, color='dimgrey', label='placa subductada')
plt.axhline(0, 0, 0.5, linewidth=14, color='dimgrey')
plt.ylim(-100, 0)
plt.axhline(0, 0.529, linewidth=14, color='forestgreen')
plt.axhline(0, 0.6, linewidth=7, color='forestgreen', label='Placa continental')
plt.title('Corriente en todo el dominio')
plt.xlabel('Distancia desde el eje de la fosa [km]')
plt.ylabel('Profundidad [km]')
plt.colorbar()
plt.legend()
#plt.savefig('contornos de corriente60grados.png')
plt.show()

plt.figure(figsize=(13, 5))
plt.contourf(X*10**-3, Z*10**-3, psi_final, levels=20, cmap='Spectral')
plt.quiver(X*10**-3, Z*10**-3, velx_final, velz_final, np.sqrt(velz_final**2+velx_final**2),
           width=0.0015, label='Vectores de velocidad')

plt.axline((0, 0), (100, 100*np.tan(beta1)), linewidth=7, color='dimgrey', label='placa subductada')
plt.axhline(0, 0, 0.5, linewidth=14, color='dimgrey')
plt.axhline(0, 0.529, linewidth=14, color='forestgreen')
plt.axhline(0, 0.6, linewidth=7, color='forestgreen', label='Placa continental')
plt.ylim(-100, 0)
plt.title('Corriente en todo el dominio con los vectores de velocidad')
plt.xlabel('Distancia desde el eje de la fosa [km]')
plt.ylabel('Profundidad [km]')
plt.colorbar()
plt.legend()
#plt.savefig('lineas de contorno con velocidad30grados.png')
plt.show()


radios = np.linspace(1, 100*10**3, 50)
radio = 50*10**3  # m
angulos = np.linspace(0, 90, 50)
betas = -angulos * np.pi/180
constantes1 = ctes1(betas)
constantes2 = ctes2(betas)
presion1 = presion(radio, betas, constantes1[2], constantes1[3])  # Pa
presion2 = presion(radio, betas, constantes2[2], constantes2[3])
pres_tot = np.abs(presion2) + np.abs(presion1)
torques = torque(pres_tot, radio) *10**-9 #  en Pa*m

plt.plot(angulos, torques)
plt.title('Gráfico ángulo de subducción vs torque para un radio de 50[km]')
plt.ylabel('torque [GNm]')
plt.xlabel('Ángulo en grados')
#plt.savefig('grafico angulo de subduccion vs torque.png')
plt.show()






