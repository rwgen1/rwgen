import sys

import numpy as np

# ---

import scipy.stats
from rwgen.rainfall import utils

x = np.array([0,0,0,0,0,1,1,2,1,2,1,2,2,3,4,4,7,14,29,100])
print(scipy.stats.skew(x))

y, r = utils.trim_array(x, 3, 10)
# print(y)
print(r, scipy.stats.skew(y))

y, r = utils.clip_array(x, 3, 10)
# print(y)
print(r, scipy.stats.skew(y))
print()

rng = np.random.default_rng()
a = rng.exponential(1, 1000)
print(scipy.stats.skew(a))
a[-1] = np.max(a) * 7
a[-2] = np.max(a) * 5
a[-3] = np.max(a) * 2.1
print(scipy.stats.skew(a))

b, r = utils.trim_array(a, 2, 10)
print(r, scipy.stats.skew(b))
# print(np.sort(b)[-10:])

b, r = utils.clip_array(a, 2, 10)
print(r, scipy.stats.skew(b))
# print(np.sort(b)[-10:])




sys.exit()

# ---

# from scipy.interpolate import RegularGridInterpolator
#
#
# def f(x, y):
#     return 2 * x**2 + 3 * y**2
#
#
# x = np.linspace(1, 4, 11)
# y = np.linspace(4, 7, 22)
# xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
# data = f(xg, yg)
#
# f1 = RegularGridInterpolator((x, y), data)
#
# pts = np.array([[2.1, 6.2], [3.3, 5.2]])
# print(f1(pts))
# print(f(2.1, 6.2), f(3.3, 5.2))
#
# sys.exit()

# ---

from rwgen.rainfall import base
from rwgen.rainfall import utils

df = utils.read_csv_('./stnsrp/output/phi.csv')

dem = utils.read_ascii_raster('./stnsrp/input/srtm_dem.asc')

# xx, yy = np.meshgrid(dem.x.values, dem.y.values)
# print(xx)
# print(yy)
# sys.exit()

for season in range(1, 12+1):
    print(season)
    interpolator, log_transformation = base.Simulator.make_phi_interpolator(df, season)

sys.exit()

# ---

from rwgen.rainfall import base
from rwgen.rainfall import utils

df = utils.read_csv_('./stnsrp/output/phi.csv')

interpolator = base.Simulator.make_phi_interpolator(df, 1)

sys.exit()

# ---

from rwgen.rainfall import utils

x = np.array([0,0,0,0,0,1,1,2,1,2,1,2,2,3,4,4,7,14,29])
y, r = utils.trimmer(x, 2, 10)
print(y)
print(r)

sys.exit()

# ---

from rwgen.rainfall import properties

phi1 = np.array([0.543226])
phi2 = np.array([0.637527])
distances = np.array([57.4543296888929])

lamda = 0.015202
beta = 0.062486
rho = 0.000460
eta = 0.971669
xi = 0.712291
gamma = 0.022540

nu = 2.0 * np.pi * rho / gamma ** 2.0

mu_1 = 1.0 / xi
mu_2 = 2.0 / xi ** 2.0
# mu_3 = 6.0 / xi ** 3.0

# rainsim_statistics = np.array([0.783749])

cc = properties.calculate_cross_correlation(
    24, 0, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi1, phi2
)

print(cc)

# ---

phi1 = np.array([0.564449])
phi2 = np.array([0.583564])
distances = np.array([26.212287])

lamda = 0.00945254898757336
beta = 0.02
rho = 0.892091280465614
eta = 0.47269953912173
xi = 1.20182852521822
gamma = 0.872755871631909

nu = 2.0 * np.pi * rho / gamma ** 2.0

mu_1 = 1.0 / xi
mu_2 = 2.0 / xi ** 2.0
# mu_3 = 6.0 / xi ** 3.0

cc = properties.calculate_cross_correlation(
    24, 0, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi1, phi2
)

print('--')
print(cc)