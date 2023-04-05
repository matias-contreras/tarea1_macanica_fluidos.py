import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from lmfit import Model




# orden teodolitos = [[automatico], [japo], aleman]
#     automatico = [azimut, elevación]
teodolitos_bloque1=[[[0, 0],
                     [344.45, 83.92],
                     [313.57, 77.39],
                     [319.40, 70.38],
                     [334.05, 69.35],
                     [343.23, 71.00],
                     [346.67, 73.73],
                     [350.25, 74.84],
                     [353.52, 77.01],
                     [358.71, 79.60],
                     [2.47, 82.20],
                     [350.54, 85.91],
                     [8.06, 86.77],
                     [17.96, 87.29],
                     [19.29, 88.82],
                     [17.52, 90.94],
                     [42.07, 93.01],
                     [17.68, 95.70],
                     [1.70, 99.99],
                     [355.72, 103.25],
                     [353.41, 105.43],
                     [351.24, 106.65],
                     [349.17, 108.47],
                     [346.95, 109.97],
                     [345.86, 111.23],
                     [345.51, 112.16],
                     [346.11, 112.85],
                     [347.13, 113.42],
                     [347.44, 113.89],
                     [347.02, 114.37],
                     [346.63, 114.89],
                     [345.40, 115.38],
                     [343.99, 115.89],
                     [342.08, 116.02],
                     [338.78, 115.91],
                     [335.26, 115.85],
                     [331.27, 115.91],
                     [327.23, 116.04],
                     [322.98, 116.18],
                     [318.89, 116.38],
                     [315.09, 116.76]],
                    [[0, 0],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [335, 72.5],
                     [339, 74.2],
                     [np.nan, np.nan],
                     [328, 69.5],
                     [340.8, 70.2],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [353.9, 80.9],
                     [352.2, 49],
                     [0.2, 71.2],
                     [356.2, 89],
                     [350.9, 85.9],
                     [11.1, 86.2],
                     [15.1, 88.3],
                     [10.9, 90.8],
                     [18.1, 91.5],
                     [4.3, 99.5],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [160.1, 56.5],
                     [160.3, 66.1],
                     [160.2, 65.4],
                     [160.1, 75],
                     [158.2, 64.6],
                     [158, 63.9],
                     [156.1, 63.6],
                     [153.1, 63.8],
                     [149.6, 63.3],
                     [146.1, 63.5],
                     [142, 63.1],
                     [138.1, 63],
                     [134.2, 62.8]],
                    [[0, 0],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [348.5, 70],
                      [354.5, 71.1],
                      [359, 76.8],
                      [2, 75.1],
                      [6.5, 76.9],
                      [12.2, 79.8],
                      [18.5, 81.9],
                      [13, 86],
                      [97, 33.1],
                      [51.5, 87],
                      [73, 88.5],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [np.nan, np.nan],
                      [177.9, 66],
                      [177.2, 65.5],
                      [177, 65],
                      [176, 64.5],
                      [174.2, 64],
                      [172.5, 63.9],
                      [169.5, 63.9],
                      [166, 64],
                      [162, 64],
                      [157.5, 64],
                      [153, 66.2],
                      [149.8, 63.5]]]



teodolitos_bloque2 =[[[0, 0],
                      [7.10, 30.70],
                      [275.57, 77.48],
                      [322.85, 82.46],
                      [308.75, 84.16],
                      [276.67, 85.76],
                      [250.71, 84.45],
                      [255.83, 84.43],
                      [249.85, 89.86],
                      [249.70, 97.50],
                      [254.27, 104.13],
                      [257.93, 108.66],
                      [264.94, 110.93],
                      [269.01, 112.82],
                      [270.69, 114.38],
                      [272.48, 114.94],
                      [277.37, 114.21],
                      [282.71, 112.15],
                      [288.54, 110.93],
                      [295.72, 110.41],
                      [305.16, 111.90],
                      [313.10, 115.15],
                      [316.94, 117.04],
                      [319.45, 118.91],
                      [321.40, 120.39],
                      [323.29, 121.44],
                      [325.46, 122.27],
                      [327.72, 123.19],
                      [330.22, 123.96],
                      [332.43, 124.63],
                      [333.99, 125.31],
                      [334.99, 125.87],
                      [335.30, 126.53],
                      [334.60, 127.24],
                      [333.21, 127.65],
                      [331.37, 128.02],
                      [328.92, 128.28],
                      [326.34, 128.58],
                      [323.89, 128.83],
                      [321.81, 129.18],
                      [319.77, 129.52],
                      [318.13, 129.94]],
                    [[0, 0],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [308.1, 85.8],
                     [285, 85.2],
                     [256.3, 84.2],
                     [258.9, 84.9],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [75, 76.5],
                     [79.1, 71.1],
                     [86.8, 69],
                     [89.6, 66.9],
                     [91.2, 65.3],
                     [93.5, 65],
                     [98.4, 65.7],
                     [104.2, 67.7],
                     [109, 68.8],
                     [117.1, 69.5],
                     [127.1, 67.7],
                     [133.2, 64.6],
                     [136.6, 62.6],
                     [138.2, 60.8],
                     [141.3, 59.5],
                     [143, 58.3],
                     [144.8, 57.7],
                     [147.9, 56.8],
                     [147.5, 56],
                     [150.5, 55.4],
                     [151.8, 54.9],
                     [153, 54.2],
                     [153, 53.5],
                     [151.9, 52.9],
                     [150.7, 52.4],
                     [148.5, 51.9],
                     [146.4, 51.7],
                     [143.9, 51.2],
                     [141.5, 51],
                     [139.6, 50.5],
                     [137.7, 50.2],
                     [136.2, 49.7]],
                    [[0, 0],
                     [np.nan, np.nan],
                     [318, 81.5],
                     [314.2, 84],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [np.nan, np.nan],
                     [67.9, 75],
                     [70.9, 71.1],
                     [88.9, 69.1],
                     [82.2, 67.1],
                     [84, 65.9],
                     [85.9, 65.1],
                     [91, 66],
                     [96.4, 68],
                     [102.4, 69.2],
                     [109.5, 64.9],
                     [119, 68],
                     [126.9, 64.9],
                     [130.8, 62.9],
                     [133, 61.2],
                     [np.nan, np.nan],
                     [136.9, 58.7],
                     [139, 66.8],
                     [141.2, 56.9],
                     [143.9, 56],
                     [146, 55.5],
                     [147.5, 54.9],
                     [148.2, 54.1],
                     [148.8, 53.6],
                     [148, 53],
                     [146.5, 52.2],
                     [144.7, 52.1],
                     [142, 51.9],
                     [139.6, 51.5],
                     [137.2, 51.1],
                     [135, 50.9],
                     [133, 50.3],
                     [131.2, 50.1]]]

auto_bloque1 = np.array(teodolitos_bloque1[0]) * np.pi / 180  # en radianes
japo_bloque1 = np.array(teodolitos_bloque1[1]) * np.pi / 180  # en radianes
ale_bloque1 = np.array(teodolitos_bloque1[2]) * np.pi / 180  # en radianes
auto_bloque2 = np.array(teodolitos_bloque2[0]) * np.pi / 180  # en radianes
japo_bloque2 = np.array(teodolitos_bloque2[1]) * np.pi / 180  # en radianes
ale_bloque2 = np.array(teodolitos_bloque2[2]) * np.pi / 180  # en radianes


b1_auto_azimut = auto_bloque1[0:41, 0]
b1_auto_elev = auto_bloque1[0:41, 1]
b1_japo_azimut = japo_bloque1[0:41, 0]
b1_japo_elev = japo_bloque1[0:41, 1]
b1_ale_azimut = ale_bloque1[0:41, 0]
b1_ale_elev = ale_bloque1[0:41, 1]

b2_auto_azimut = auto_bloque2[0:41, 0]
b2_auto_elev = auto_bloque2[0:41, 1]
b2_japo_azimut = japo_bloque2[0:41, 0]
b2_japo_elev = japo_bloque2[0:41, 1]
b2_ale_azimut = ale_bloque2[0:41, 0]
b2_ale_elev = ale_bloque2[0:41, 1]

tiempo = np.arange(0, 30*41, 30)


def correccion_data(dataset):
    """
    corrige los angulos altos, los cercanos a 360 grados, los pasa a angulos negativos
    :param dataset: vector con angulos azimutales
    :return: vector con angulos en torno al cero
    """
    data_corr = []
    for i in dataset:
        if i >= 1.2 * np.pi:
            i = i - 2 * np.pi
        data_corr.append(i)
    return data_corr


def gaussian(x, a, x0, s, a1, x1, s1, m, n):

    amp = a/np.sqrt(2 * np.pi * s**2)
    exp = -1 * (x - x0)**2 / (2 * s**2)
    gauss = amp * np.exp(exp)
    amp1 = a1 / np.sqrt(2 * np.pi * s1 ** 2)
    exp1 = -1 * (x - x1) ** 2 / (2 * s1 ** 2)
    gauss1 = amp1 * np.exp(exp1)
    y = m * x + n
    return gauss + gauss1 + y


def gaussian_para(x, a, x0, s, a1, x1, s1, m, n, p1, p2, p3):

    amp = a/np.sqrt(2 * np.pi * s**2)
    exp = -1 * (x - x0)**2 / (2 * s**2)
    gauss = amp * np.exp(exp)
    amp1 = a1 / np.sqrt(2 * np.pi * s1 ** 2)
    exp1 = -1 * (x - x1) ** 2 / (2 * s1 ** 2)
    gauss1 = amp1 * np.exp(exp1)
    y = m * x + n
    parabola = p1 * x**2 + p2 * x + p3
    return gauss + gauss1 + y + parabola

def gaussian_exp(x, a, x0, s, a1, x1, s1, m, n, y0):

    amp = a/np.sqrt(2 * np.pi * s**2)
    exp = -1 * (x - x0)**2 / (2 * s**2)
    gauss = amp * np.exp(exp)
    amp1 = a1 / np.sqrt(2 * np.pi * s1 ** 2)
    exp1 = -1 * (x - x1) ** 2 / (2 * s1 ** 2)
    gauss1 = amp1 * np.exp(exp1)
    y = m * x + n
    return gauss + gauss1 + y + y0


def x_este(tiempo, azimut, elev):
    """
    :param tiempo: vector con los tiempos
    :param azimut: vector con los angulos azimutales en radianes
    :param elev: vector con los angulos de elevacion en radianes
    :return: posicion en la orientación este-oeste
    """
    pos = 3 * tiempo * np.sin(azimut) / np.tan(elev)
    return pos


def x_norte(tiempo, azimut, elev):
    """
        :param tiempo: vector con los tiempos
        :param azimut: vector con los angulos azimutales en radianes
        :param elev: vector con los angulos de elevacion en radianes
        :return: posicion en la orientación norte-sur
        """
    pos = 3 * tiempo * np.cos(azimut) / np.tan(elev)
    return pos


def altura(tiempo):
    """
    encuentra la altura, considerando una tasa de ascenso del globo de 3 metros/seg
    :param tiempo: vector con los tiempos
    :return: altura en metros
    """
    return 3 * tiempo


def velocidad(di, df, dt):
    v = (df - di) / dt
    return v

def velo_globo(posiciones, tiempo):
    velo = []
    for i in range(0, len(posiciones)-1):
        dt = tiempo[i+1] - tiempo[i]
        vel = velocidad(posiciones[i], posiciones[i+1], dt)
        velo.append(vel)
    return np.array(velo)

def velocidad_vector(velo_norte, velo_este):
    mag = np.sqrt(velo_norte**2 + velo_este**2)
    theta = np.arctan(velo_este / velo_norte) * 180 / np.pi #para que quede en grados
    velos = [mag, theta]
    return velos



b1_auto_azimut_corr = correccion_data(b1_auto_azimut)
b1_japo_azimut_corr = correccion_data(b1_japo_azimut)
b1_ale_azimut_corr = correccion_data(b1_ale_azimut)

b2_auto_azimut_corr = correccion_data(b2_auto_azimut)
b2_japo_azimut_corr = correccion_data(b2_japo_azimut)
b2_ale_azimut_corr = correccion_data(b2_ale_azimut)


"""
=============================================0
se eliminan los valores NaN de cada vector de azimut y elevacion de
las mediciones del japo y aleman del bloque 1
"""

ii_b1_japo = np.isfinite(b1_japo_azimut_corr)
tiempo_b1_japo =[]
corto_b1_japo_azimut=[]
corto_b1_japo_elev = []
for i in range(0,len(b1_japo_azimut_corr)):
    if ii_b1_japo[i] == True:
        tiempo_b1_japo.append(tiempo[i])
        corto_b1_japo_azimut.append(b1_japo_azimut_corr[i])
        corto_b1_japo_elev.append(b1_japo_elev[i])


ii_b1_ale = np.isfinite(b1_ale_azimut_corr)
tiempo_b1_ale =[]
corto_b1_ale_azimut=[]
corto_b1_ale_elev = []
for i in range(0,len(b1_ale_azimut_corr)):
    if ii_b1_ale[i] == True:
        tiempo_b1_ale.append(tiempo[i])
        corto_b1_ale_azimut.append(b1_ale_azimut_corr[i])
        corto_b1_ale_elev.append(b1_ale_elev[i])




"""
=============================================0
se eliminan los valores NaN de cada vector de azimut y elevacion de
las mediciones del japo y aleman del bloque 2
las del bloque 1 se hicieron al ojo.
"""


ii_b2_japo = np.isfinite(b2_japo_azimut_corr)
tiempo_b2_japo =[]
corto_b2_japo_azimut=[]
corto_b2_japo_elev = []
for i in range(0,len(b2_japo_azimut_corr)):
    if ii_b2_japo[i] == True:
        tiempo_b2_japo.append(tiempo[i])
        corto_b2_japo_azimut.append(b2_japo_azimut_corr[i])
        corto_b2_japo_elev.append(b2_japo_elev[i])


ii_b2_ale = np.isfinite(b2_ale_azimut_corr)
tiempo_b2_ale =[]
corto_b2_ale_azimut=[]
corto_b2_ale_elev = []
for i in range(0,len(b2_ale_azimut_corr)):
    if ii_b2_ale[i] == True:
        tiempo_b2_ale.append(tiempo[i])
        corto_b2_ale_azimut.append(b2_ale_azimut_corr[i])
        corto_b2_ale_elev.append(b2_ale_elev[i])





# se hace el curve fit para el automatico del bloque 1 con gaussian

initial_b1_auto_azimut = [ 4.91539431e+02,  4.51577953e+02, -1.77799897e+02,  150,
        900,  100, -9.18548654e-05,  4.25922958e+02,
       -4.26670441e+02]
popt_b1_auto_azimut, pcov_b1_auto_azimut = curve_fit(gaussian_exp, tiempo[2:41], b1_auto_azimut_corr[2:41],
                       p0=initial_b1_auto_azimut)

azimut_b1_auto_final = gaussian_exp(tiempo, *popt_b1_auto_azimut)


initial_b1_auto_elev = [20, 40,  10,
                        3300, 780,  950,
                        6.60614454e-04,  1.21433603e+00, -1.25]

popt_b1_auto_elev, pcov_b1_auto_elev = curve_fit(gaussian_exp, tiempo[2:41],
                                                     b1_auto_elev[2:41],
                                                     p0=initial_b1_auto_elev)

elev_b1_auto_final = gaussian_exp(tiempo, *popt_b1_auto_elev)


# se hace el curve fit para el japones del bloque 1 con gaussian_para


initial_b1_japo_azimut = [-4.17788197e+02, 6.86181843e+01, 1.47284544e+02,
                          920, 750, 500,
                          -2.06731029e-03, 5.19013462e-01, 4.18219458e-06,
                          -2.06707997e-03, 5.13468602e-01]

popt_b1_japo_azimut, pcov_b1_japo_azimut = curve_fit(gaussian_para, tiempo_b1_japo,
                                                     corto_b1_japo_azimut,
                                                     p0=initial_b1_japo_azimut)

azimut_b1_japo_final = gaussian_para(tiempo, *popt_b1_japo_azimut)


initial_b1_japo_elev = [500, 150, 100, 800, 600, 400, 0, 0]

popt_b1_japo_elev, pcov_b1_japo_elev = curve_fit(gaussian, tiempo_b1_japo,
                                                     corto_b1_japo_elev,
                                                     p0=initial_b1_japo_elev)

elev_b1_japo_final = gaussian(tiempo, *popt_b1_japo_elev)


# se hace el curve fit para el aleman del bloque 1 con gaussian_para

initial_b1_ale_azimut = [-360,  60,  100,
                         1200, 8.85880251e+02,  200,
                         -2.06731029e-03,  5.19013462e-01, 4.18219458e-06,
                         -2.06707997e-03,  5.13468602e-01]

popt_b1_ale_azimut, pcov_b1_ale_azimut = curve_fit(gaussian_para, tiempo_b1_ale,
                                                     corto_b1_ale_azimut,
                                                     p0=initial_b1_ale_azimut)

azimut_b1_ale_final = gaussian_para(tiempo, *popt_b1_ale_azimut)

initial_b1_ale_elev = [-100, 30, 100, 1600, 500, 400, 0, 0]

tiempo_b1_ale.pop(9)
corto_b1_ale_elev.pop(9)


popt_b1_ale_elev, pcov_b1_ale_elev = curve_fit(gaussian, tiempo_b1_ale,
                                                     corto_b1_ale_elev,
                                                     p0=initial_b1_ale_elev)
elev_b1_ale_final = gaussian(tiempo, *popt_b1_ale_elev)


#plt.figure()
#plt.plot(tiempo, b1_auto_elev, '*')
#plt.plot(tiempo, elev_b1_auto_final, '-+')
#plt.show()




# se hace el curve fit para el automatico del bloque 2 con gaussian

initial_b2_auto_azimut = [-200, 200, 200, 100, 980, 300, 0, 0]
popt_b2_auto_azimut, pcov_b2_auto_azimut = curve_fit(gaussian, tiempo, b2_auto_azimut_corr,
                       p0=initial_b2_auto_azimut)

azimut_b2_auto_final = gaussian(tiempo, *popt_b2_auto_azimut)


# se fitea los datos de elevacion del auto del bloque 2 con gaussian_para

initial_b2_auto_elev = [130, 75, 53, 110, 200, 250, 360, -3*10**5, -3*10**-6, -360, 3*10**5]
popt_b2_auto_elev, pcov_b2_auto_elev = curve_fit(gaussian_para, tiempo, b2_auto_elev,
                                                 p0=initial_b2_auto_elev)

elev_b2_auto_final = gaussian_para(tiempo, *popt_b2_auto_elev)


#se fitean los datos del japo bloque 2 azimut con gausian_para

initial_b2_japo_azimut = [-350, 170, 55, 3200, 900, 500, 0, 0, 0, 0, 0]

popt_b2_japo_azimut, pcov_b2_japo_azimut = curve_fit(gaussian_para, tiempo_b2_japo,
                                                     corto_b2_japo_azimut,
                                                     p0=initial_b2_japo_azimut)

azimut_b2_japo_final = gaussian_para(tiempo, *popt_b2_japo_azimut)


#se fitea la elevacion para el japo bloque 2

initial_b2_japo_elev = [400, 170, 200, 200, 550, 200, 0, 0]

popt_b2_japo_elev, pcov_b2_japo_elev = curve_fit(gaussian, tiempo_b2_japo[1:30],
                                                 corto_b2_japo_elev[1:30],
                                                 p0=initial_b2_japo_elev)

elev_b2_japo_final = gaussian(tiempo, *popt_b2_japo_elev)


# se fitean los datos del azimut aparato aleman tomado en el bloque 2

initial_b2_ale_azimut = [-350, 170, 55, 3200, 900, 500, 0, 0, 0, 0, 0]

popt_b2_ale_azimut, pcov_b2_ale_azimut = curve_fit(gaussian_para, tiempo_b2_ale,
                                                   corto_b2_ale_azimut,
                                                   p0=initial_b2_ale_azimut)

azimut_b2_ale_final = gaussian_para(tiempo, *popt_b2_ale_azimut)

# se fitean los datos de elevacion del aleman bloque 2

initial_b2_ale_elev = [400, 170, 200, 200, 550, 200, 0, 1.4]

popt_b2_ale_elev, pcov_b2_ale_elev = curve_fit(gaussian, tiempo_b2_ale[1:25],
                                                 corto_b2_ale_elev[1:25],
                                                 p0=initial_b2_ale_elev)

elev_b2_ale_final = gaussian(tiempo, *popt_b2_ale_elev)










b1_auto_x_este = x_este(tiempo, azimut_b1_auto_final, elev_b1_auto_final)
b1_auto_x_norte = x_norte(tiempo, azimut_b1_auto_final, elev_b1_auto_final)

b1_japo_x_este = x_este(tiempo, azimut_b1_japo_final, elev_b1_japo_final)
b1_japo_x_norte = x_norte(tiempo, azimut_b1_japo_final, elev_b1_japo_final)

b1_ale_x_este = x_este(tiempo, azimut_b1_ale_final, elev_b1_ale_final)
b1_ale_x_norte = x_norte(tiempo, azimut_b1_ale_final, elev_b1_ale_final)


b2_auto_x_este = x_este(tiempo, azimut_b2_auto_final, elev_b2_auto_final)
b2_auto_x_norte = x_norte(tiempo, azimut_b2_auto_final, elev_b2_auto_final)

b2_japo_x_este = x_este(tiempo, azimut_b2_japo_final, elev_b2_japo_final)
b2_japo_x_norte = x_norte(tiempo, azimut_b2_japo_final, elev_b2_japo_final)

b2_ale_x_este = x_este(tiempo, azimut_b2_ale_final, elev_b2_ale_final)
b2_ale_x_norte = x_norte(tiempo, azimut_b2_ale_final, elev_b2_ale_final)


# calculamos las velocidades para cada intervalo de tiempo


vel_b1_auto_este = velo_globo(b1_auto_x_este, tiempo)
vel_b1_auto_norte = velo_globo(b1_auto_x_norte, tiempo)

vel_b1_japo_este = velo_globo(b1_japo_x_este, tiempo)
vel_b1_japo_norte = velo_globo(b1_japo_x_norte, tiempo)

vel_b1_ale_este = velo_globo(b1_ale_x_este, tiempo)
vel_b1_ale_norte = velo_globo(b1_ale_x_norte, tiempo)


vel_b2_auto_este = velo_globo(b2_auto_x_este, tiempo)
vel_b2_auto_norte = velo_globo(b2_auto_x_norte, tiempo)

vel_b2_japo_este = velo_globo(b2_japo_x_este, tiempo)
vel_b2_japo_norte = velo_globo(b2_japo_x_norte, tiempo)

vel_b2_ale_este = velo_globo(b2_ale_x_este, tiempo)
vel_b2_ale_norte = velo_globo(b2_ale_x_norte, tiempo)

# velocidades expresadas en su magnitud y en el angulo que forma la dirección del viento
# con el norte, siendo un viento sur cero grados

velos_b1_auto = velocidad_vector(vel_b1_auto_norte, vel_b1_auto_este)
velos_b1_japo = velocidad_vector(vel_b1_japo_norte, vel_b1_japo_este)
velos_b1_ale = velocidad_vector(vel_b1_ale_norte, vel_b1_ale_este)

velos_b2_auto = velocidad_vector(vel_b2_auto_norte, vel_b2_auto_este)
velos_b2_japo = velocidad_vector(vel_b2_japo_norte, vel_b2_japo_este)
velos_b2_ale = velocidad_vector(vel_b2_ale_norte, vel_b2_ale_este)



plt.figure(1)
plt.title('Gráfico posición medido por los teodolitos desde las 10:57 am')
plt.xlabel('Posición este-oeste [metros]')
plt.ylabel('posición norte-sur [metros]')
plt.plot(b2_auto_x_este, b2_auto_x_norte, label='Automático')
plt.plot(b2_japo_x_este, b2_japo_x_norte, label='Japones manual')
plt.plot(b2_ale_x_este, b2_ale_x_norte, label='Aleman manual')
plt.legend()
plt.savefig('tareas/posicion_b2.png')
plt.show()


plt.figure(2)
plt.title('Gráfico posición medido por los teodolitos desde las 9:39 am')
plt.xlabel('Posición este-oeste [metros]')
plt.ylabel('posición norte-sur [metros]')
plt.plot(b1_auto_x_este, b1_auto_x_norte, label='Automático')
plt.plot(b1_japo_x_este, b1_japo_x_norte, label='Japones manual')
plt.plot(b1_ale_x_este, b1_ale_x_norte, label='Aleman manual')
plt.legend()
plt.savefig('tareas/posicion_b1.png')
plt.show()
"""

plt.figure(3, figsize=(10,6))
plt.title('Ángulos azimutales del teodolito automático del bloque de las 9:39 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b1_auto_azimut_corr, '*', label='azimuth medidos')
plt.plot(tiempo, azimut_b1_auto_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/azimut_b1_auto.png')
plt.show()


plt.figure(4, figsize=(10,6))
plt.title('Ángulos de elevación del teodolito automático del bloque de las 9:39 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b1_auto_elev, '*', label='azimuth medidos')
plt.plot(tiempo, elev_b1_auto_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/elev_b1_auto.png')
plt.show()


plt.figure(5, figsize=(10,6))
plt.title('Ángulos azimutales del teodolito japonés del bloque de las 9:39 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b1_japo_azimut_corr, '*', label='azimuth medidos')
plt.plot(tiempo, azimut_b1_japo_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/azimut_b1_japo.png')
plt.show()


plt.figure(6, figsize=(10,6))
plt.title('Ángulos de elevación del teodolito japonés del bloque de las 9:39 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b1_japo_elev, '*', label='azimuth medidos')
plt.plot(tiempo, elev_b1_japo_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/elev_b1_japo.png')
plt.show()


plt.figure(7, figsize=(10,6))
plt.title('Ángulos azimutales del teodolito aleman del bloque de las 9:39 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b1_ale_azimut_corr, '*', label='azimuth medidos')
plt.plot(tiempo, azimut_b1_ale_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/azimut_b1_ale.png')
plt.show()


plt.figure(8, figsize=(10,6))
plt.title('Ángulos de elevación del teodolito aleman del bloque de las 9:39 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b1_ale_elev, '*', label='azimuth medidos')
plt.plot(tiempo, elev_b1_ale_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/elev_b1_ale.png')
plt.show()


plt.figure(9, figsize=(10,6))
plt.title('Ángulos azimutales del teodolito automático del bloque de las 10:57 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b2_auto_azimut_corr, '*', label='azimuth medidos')
plt.plot(tiempo, azimut_b2_auto_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/azimut_b2_auto.png')
plt.show()


plt.figure(10, figsize=(10,6))
plt.title('Ángulos de elevación del teodolito automático del bloque de las 10:57 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b2_auto_elev, '*', label='azimuth medidos')
plt.plot(tiempo, elev_b2_auto_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/elev_b2_auto.png')
plt.show()


plt.figure(11, figsize=(10,6))
plt.title('Ángulos azimutales del teodolito japonés del bloque de las 10:57 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b2_japo_azimut_corr, '*', label='azimuth medidos')
plt.plot(tiempo, azimut_b2_japo_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/azimut_b2_japo.png')
plt.show()


plt.figure(12, figsize=(10,6))
plt.title('Ángulos de elevación del teodolito japonés del bloque de las 10:57 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b2_japo_elev, '*', label='azimuth medidos')
plt.plot(tiempo, elev_b2_japo_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/elev_b2_japo.png')
plt.show()


plt.figure(13, figsize=(10,6))
plt.title('Ángulos azimutales del teodolito aleman del bloque de las 10:57 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b2_ale_azimut_corr, '*', label='azimuth medidos')
plt.plot(tiempo, azimut_b2_ale_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/azimut_b2_ale.png')
plt.show()


plt.figure(14, figsize=(10,6))
plt.title('Ángulos de elevación del teodolito aleman del bloque de las 10:57 am')
plt.xlabel('tiempo [segundos]')
plt.ylabel(r' $\theta$ [rad]')
plt.plot(tiempo, b2_ale_elev, '*', label='azimuth medidos')
plt.plot(tiempo, elev_b2_ale_final, '-+', label='suavización')
plt.legend()
plt.savefig('tareas/elev_b2_ale.png')
plt.show()


origen_b1_ale = np.array([velos_b1_ale[0], 3*tiempo[1:41]])
plt.figure(15, figsize=[10,8])
plt.plot(velos_b1_ale[0], 3*tiempo[1:41], '*')
plt.quiver(*origen_b1_ale, vel_b1_ale_este, vel_b1_ale_norte, scale=0.01, scale_units='y', width=0.002, label='dirección del viento')
plt.xlim(0, 6.3)
plt.grid()
plt.title('Perfil de altura con magnitud y dirección de velocidad del viento medido por el teodolito aleman desde la 9:39 am')
plt.xlabel(r' Magnitud de la velocidad del viento $\left[\frac{m}{s}\right]$')
plt.ylabel('Altura [m]')
plt.legend()
plt.savefig('tareas/velos_b1_ale.png')
plt.show()


origen_b1_japo = np.array([velos_b1_japo[0], 3*tiempo[1:41]])
plt.figure(16, figsize=[10,8])
plt.plot(velos_b1_japo[0], 3*tiempo[1:41], '*')
plt.quiver(*origen_b1_japo, vel_b1_japo_este, vel_b1_japo_norte, scale=0.01, scale_units='y', width=0.002, label='dirección del viento')
plt.xlim(0, 7)
plt.ylim(0, 3800)
plt.grid()
plt.title('Perfil de altura con magnitud y dirección de velocidad del viento medido por el teodolito japonés desde la 9:39 am')
plt.xlabel(r' Magnitud de la velocidad del viento $\left[\frac{m}{s}\right]$')
plt.ylabel('Altura [m]')
plt.legend()
plt.savefig('tareas/velos_b1_japo.png')
plt.show()


origen_b1_auto = np.array([velos_b1_auto[0], 3*tiempo[1:41]])
plt.figure(17, figsize=[10,8])
plt.plot(velos_b1_auto[0], 3*tiempo[1:41], '*')
plt.quiver(*origen_b1_auto, vel_b1_auto_este, vel_b1_auto_norte, scale=0.01, scale_units='y', width=0.002, label='dirección del viento')
plt.xlim(0, 5)
plt.grid()
plt.title('Perfil de altura con magnitud y dirección de velocidad del viento medido por el teodolito automático desde la 9:39 am')
plt.xlabel(r' Magnitud de la velocidad del viento $\left[\frac{m}{s}\right]$')
plt.ylabel('Altura [m]')
plt.legend()
plt.savefig('tareas/velos_b1_auto.png')
plt.show()


origen_b2_auto = np.array([velos_b2_auto[0], 3*tiempo[1:41]])
plt.figure(18, figsize=[10,8])
plt.plot(velos_b2_auto[0], 3*tiempo[1:41], '*')
plt.quiver(*origen_b2_auto, vel_b2_auto_este, vel_b2_auto_norte, scale=0.01, scale_units='y', width=0.002, label='dirección del viento')
plt.xlim(0, 6)
plt.ylim(0, 4200)
plt.grid()
plt.title('Perfil de altura con magnitud y dirección de velocidad del viento medido por el teodolito automático desde la 10:57 am')
plt.xlabel(r' Magnitud de la velocidad del viento $\left[\frac{m}{s}\right]$')
plt.ylabel('Altura [m]')
plt.legend()
plt.savefig('tareas/velos_b2_auto.png')
plt.show()


origen_b2_japo = np.array([velos_b2_japo[0], 3*tiempo[1:41]])
plt.figure(19, figsize=[10,8])
plt.plot(velos_b2_japo[0], 3*tiempo[1:41], '*')
plt.quiver(*origen_b2_japo, vel_b2_japo_este, vel_b2_japo_norte, scale=0.01, scale_units='y', width=0.002, label='dirección del viento')
plt.xlim(0, 6)
plt.grid()
plt.title('Perfil de altura con magnitud y dirección de velocidad del viento medido por el teodolito japonés desde la 10:57 am')
plt.xlabel(r' Magnitud de la velocidad del viento $\left[\frac{m}{s}\right]$')
plt.ylabel('Altura [m]')
plt.legend()
plt.savefig('tareas/velos_b2_japo.png')
plt.show()


origen_b2_ale = np.array([velos_b2_ale[0], 3*tiempo[1:41]])
plt.figure(20, figsize=[10,8])
plt.plot(velos_b2_ale[0], 3*tiempo[1:41], '*')
plt.quiver(*origen_b2_ale, vel_b2_ale_este, vel_b2_ale_norte, scale=0.01, scale_units='y', width=0.002, label='dirección del viento')
plt.xlim(0, 9.5)
plt.grid()
plt.title('Perfil de altura con magnitud y dirección de velocidad del viento medido por el teodolito aleman desde la 10:57 am')
plt.xlabel(r' Magnitud de la velocidad del viento $\left[\frac{m}{s}\right]$')
plt.ylabel('Altura [m]')
plt.legend()
plt.savefig('tareas/velos_b2_ale.png')
plt.show()
"""