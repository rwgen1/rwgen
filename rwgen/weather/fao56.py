"""
FAO56 (Allen et al. 1998) reference evapotranspiration functions.

"""
import datetime
import calendar

import numpy as np


def atmos_pressure(z):
    """ Estimate atmospheric pressure based on elevation (equation 7).
    :param z: elevation [m]
    :return p: atmospheric pressure [kPa degC-1]
    """
    p = 101.3 * (((293.0 - (0.0065 * z)) / 293.0) ** 5.26)
    return p


def delta_svp(t):
    """ Calculate the slope of the vapour pressure curve (equation 13).
    :param t: temperature [deg K]
    :return delta: slope of vapour pressure curve [kPa degC-1]
    """
    delta = (
        (4098.0 * (0.6108 * np.exp((17.27 * (t - 273.15)) / ((t - 273.15) + 237.3)))) / (((t - 273.15) + 237.3) ** 2)
    )
    return delta


def psy_const(p):
    """ Calculate the psychrometric constant (equation 8).
    :param p: pressure [kPa]
    :return: psychrometric constant [kPa degC-1]
    """
    return 0.000665 * p


def windspeed_2m(ws, z):
    """ Adjust wind measurement height to 2m (equation 47).
    :param ws: wind speed [m s-1]
    :param z: measurement height [m]
    :return: wind speed at 2m [m s-1]
    """
    return ws * (4.87 / (np.log((67.8 * float(z)) - 5.42)))


def sat_vap_press(t):
    """ Estimate saturation vapour pressure from temperature (equation 11).
    :param t: temperature [degK]
    :return svp: saturation vapour pressure [kPa]
    """
    svp = 0.6108 * np.exp((17.27 * (t - 273.15)) / ((t - 273.15) + 237.3))
    return svp


def mean_svp(tmin, tmax):
    """ Estimate daily mean saturation vapour pressure from Tmin and Tmax
    (equation 12).
    :param tmin: minimum temperature [degK]
    :param tmax: maximum temperature [degK]
    :return svp: mean saturation vapour pressure [kPa]
    """
    svp_tmin = sat_vap_press(tmin)
    svp_tmax = sat_vap_press(tmax)
    svp = (svp_tmin + svp_tmax) / 2.0
    return svp


def avp_from_tdew(tdew):
    """ Estimate actual vapour pressure from dewpoint temperature
    (equation 14).
    :param tdew: temperature [degK]
    :return avp: actual vapour pressure [kPa]
    """
    avp = 0.6108 * np.exp((17.27 * (tdew - 273.15)) / ((tdew - 273.15) + 237.3))
    return avp


def avp_from_rh(tmin, tmax, rhmin, rhmax):
    """ Estimate actual vapour pressure from relative humidity (equation 17).
    :param tmin: day minimum temperature [degK]
    :param tmax: day maximum temperature [degK]
    :param rhmin: day minimum relative humidity [%]
    :param rhmax: day maximum relative humidity [%]
    :return avp: actual vapour pressure [kPa]
    """
    svp_tmin = sat_vap_press(tmin)
    svp_tmax = sat_vap_press(tmax)
    avp = ((svp_tmin * (rhmax / 100.0)) + (svp_tmax * (rhmin / 100.0))) / 2.0
    return avp


def fao56_et0(delta_svp, netrad, shf, psy, t, ws2, svp, avp):
    """ Calculate ET0 (equation 6).
    :param delta_svp: slope of vapour pressure curve [kPa degC-1]
    :param netrad: net radiation [MJ m-2 day-1]
    :param shf: soil heat flux [MJ m-2 day-1]
    :param psy: psychrometric constant [kPa degC-1]
    :param t: air temperature [degK]
    :param ws2: 2m wind speed [m s-1]
    :param svp: saturation vapour pressure [kPa]
    :param avp: actual vapour pressure [kPa]
    :return et0: reference evapotranspiration [mm day-1]
    """
    et0 = (
        ((0.408 * delta_svp * (netrad - shf)) + (psy * (900.0 / t) * ws2 * (svp - avp)))
        / (delta_svp + (psy * (1.0 + (0.34 * ws2))))
    )
    return et0


def extraterrestrial_radiation(dr, omega, lat, dec):
    """ Extra-terrestrial radiation (equation 21).
    :param dr: inverse relative distance Earth-Sun [-]
    :param omega: sunset hour angle [rad]
    :param lat: latitude [rad]
    :param dec: solar declination [rad]
    :return ra: extraterrestrial radiation [MJ m-2 day-1]
    """
    ra = (
        ((24.0 * 60.0) / np.pi) * 0.0820 * dr
        * ((omega * np.sin(lat) * np.sin(dec)) + (np.cos(lat) * np.cos(dec) * np.sin(omega)))
    )
    return ra


def earth_sun_distance(doy):
    """ Inverse relative distance Earth-Sun (equation 23).
    :param doy: day of year (1-365 or 1-366)
    :return dr: inverse relative distance Earth-Sun [-]
    """
    dr = 1.0 + (0.033 * np.cos(((2.0 * np.pi) / 365.0) * doy))
    return dr


def solar_declination(doy):
    """ Solar declination (equation 24).
    :param doy: day of year (1-365 or 1-366)
    :return dec: solar declination [rad]
    """
    dec = 0.409 * np.sin((((2.0 * np.pi) / 365.0) * doy) - 1.39)
    return dec


def omega_(lat, dec):
    """ Sunset hour angle (equation 25).
    :param lat: latitude [rad]
    :param dec: solar declination [rad]
    :return sunset hour angle [rad]
    """
    om = np.arccos(-1 * np.tan(lat) * np.tan(dec))
    return om


def daylight_hours(omega):
    """ Number of daylight hours under clear-sky conditions (equation 34).
    :param omega: sunset hour angle [rad]
    :return N: daylight hours [hrs]
    """
    N = 24 / np.pi * omega
    return N


def solar_radiation(ra, n, N, a=0.25, b=0.5):
    """ Downwards all-sky solar radiation at the surface using sunshine hours (equation 35).
    :param ra: extra-terrestrial radiation [MJ m-2 day-1]
    :param n: actual sunshine hours [hrs]
    :param N: potential (clear-sky) sunshine hours [hrs]
    :param a: parameter for fraction of ``ra`` reaching surface on overcast days [-]
    :param b: when added to parameter ``a`` gives the fraction of ``ra`` reaching surface on clear days [-]
    :return rs: all-sky solar radiation at the surface [MJ m-2 day-1]
    """
    # Clause to check that actual sunshine hours do not exceed potential sunshine hours
    n_ = np.minimum(n, N)
    rs = (a + b * (n_ / N)) * ra
    return rs


def clear_sky_solar_radiation(z, ra):
    """ Downwards clear-sky solar radiation at the surface (equation 37).

    Note that equation 36 could be used if data available for calibration of coefficient(s).

    :param z: elevation [m]
    :param ra: extra-terrestrial radiation [MJ m-2 day-1]
    :return rso: downwards clear-sky solar radiation at the surface [MJ m-2 day-1]
    """
    rso = (0.75 + (0.00002 * z)) * ra
    return rso


def net_solar_radiation(rs, alpha=0.23):
    """ Net downwards shortwave radiation (equation 38).

    Note that this term is positive downwards (towards the surface).

    :param rs: downwards solar radiation at the surface [MJ m-2 day-1]
    :param alpha: albedo [-]
    :return rns: net downwards shortwave radiation [MJ m-2 day-1]
    """
    rns = (1.0 - alpha) * rs
    return rns


def net_longwave_radiation(tmin, tmax, avp, rs, rso):
    """ Net upwards longwave radiation (equation 39).

    Note that this term is positive upwards (away from the surface).

    :param tmin: daily minimum temperature [K]
    :param tmax: daily maximum temperature [K]
    :param avp: actual vapour pressure [kPa]
    :param rs: downwards solar radiation at the surface [MJ m-2 day-1]
    :param rso: downwards clear-sky solar radiation at the surface [MJ m-2 day-1]
    :return rnl: net upwards longwave radiation [MJ m-2 day-1]
    """
    rnl = (
        (4.903 * 10 ** -9) * ((tmin ** 4.0 + tmax ** 4.0) / 2.0) * (0.34 - (0.14 * (avp ** 0.50)))
        * ((1.35 * (rs / rso)) - 0.35)
    )
    return rnl


def net_radiation(rns, rnl):
    """ Net downwards radiation (equation 40).
    :param rns: net downwards shortwave radiation [MJ m-2 day-1]
    :param rnl: net upwards longwave radiation [MJ m-2 day-1]
    :return rn: net downwards radiation [MJ m-2 day-1]
    """
    rn = rns - rnl
    return rn


def subdaily_extraterrestrial_radiation(doy, dt, t, dr, dec, lat, Lm, Lz):
    """ Extra-terrestrial radiation for hourly or shorter periods (equation 28).
    :param doy: day of year (1-365 or 1-366) [-]
    :param dt: timestep length [hrs]
    :param t: midpoint time of period (e.g. 14.5 for 14:00-15:00 period) [hr]
    :param dr: inverse relative distance Earth-Sun [-]
    :param dec: solar declination [rad]
    :param lat: latitude [rad]
    :param Lm: longitude of site/area in degrees WEST from Greenwich [deg]
    :param Lz: longitude of centre of local time zone in degrees WEST from Greenwich [deg]
    :param with_negative: flag to return a version of ``ra`` that includes imaginary negative values [bool]
    :return ra: extra-terrestrial radiation [MJ m-2 hr-1]
    """
    # Seasonal correction for solar time
    b = 2 * np.pi * (doy - 81) / 364  # equation 33
    Sc = 0.1645 * np.sin(2.0 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)  # equation 32

    # Solar time angle at midpoint t of period (t1, t2)
    omega_t = np.pi / 12.0 * ((t + 0.06667 * (Lz - Lm) + Sc) - 12)  # equation 31

    # Solar time angles at beginning and end of period (t1, t2)
    omega_t1 = omega_t - np.pi * dt / 24.0  # equation 29
    omega_t2 = omega_t + np.pi * dt / 24.0  # equation 30

    # Extra-terrestrial radiation - equation 28
    ra = (
        12 * 60 / np.pi * 0.0820 * dr * ((omega_t2 - omega_t1) * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec)
                                         * (np.sin(omega_t2) - np.sin(omega_t1)))
    )
    ra_all = ra.copy()

    # Limit ra to non-zero values
    ra = np.where(ra < 0.0, 0.0, ra)

    # omega_s = omega(lat, dec)  # ! could be used to check when ra should be zero - just go off negative ra though? !

    return ra
