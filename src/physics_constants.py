#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: physics_constants.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains physics constants
"""

import numpy as np

# 
R_EARTH   = 6.371e6      # Earth's mean volume radius [m]
REQ_EARTH = 6.378137e6 # Earth's equatorial radius [m]
RPL_EARTH = 6.356752e6 # Earth's polar radius [m]
ECC_EARTH = 0.003352   # Eccentricity of Earth []

M_EARTH = 5.9722e24 # Earth's mass [kg]

W_EARTH = 7.2921e-5 # Earth's angular frequency [rad/s]

GRAV_CONST = 6.674e-11 # Gravitational constant [m^3 kg^-1 s^-2]

GRAV_SURF = 9.81 # Gravitational acceleration at Earth's surface [m s^-2]

M_VEHICLE = 5498.22 # Mass of vehicle [kg]
S_VEHICLE = 12.017  # Cross sectional area of vehicle [m^2]

CD_VEHICLE = 1.2569
CS_VEHICLE = 0.0
CL_VEHICLE = 0.40815
CW_VEHICLE = np.array([CD_VEHICLE, CS_VEHICLE, CL_VEHICLE])
#CW_VEHICLE = np.array([CS_VEHICLE, CD_VEHICLE, CL_VEHICLE])

rho0 = 1.225 # Standard atmosphere reference density [kg m^-3]
H0   = 1.0e3 / 0.14 # Standard atmosphere reference height [m]

KFT_TO_KM = 0.3048


