#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: ecr2lla.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transforms from the ECR to latitude, longitude, altitude
"""

import numpy as np

from src import physics_constants as const

# 
def ecr2lla(r_ecr_, bool_is_geodetic_=False):
   """Convert position from from ECR coordinates to LLA

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    """
   
   if(1 == len(r_ecr_.shape)):
      r_ecr_ = r_ecr_[:,np.newaxis]

   longitude = np.rad2deg(np.arctan2(r_ecr_[1,:], r_ecr_[0,:])) # (y,x)
   latitude = np.rad2deg(np.arctan2(r_ecr_[2,:], np.linalg.vector_norm(r_ecr_[0:1,:], axis=0))) # (y,x)
   altitude = np.linalg.vector_norm(r_ecr_,axis=0) - const.R_EARTH

   return np.array([latitude, longitude, altitude])


'''   
   if(True == bool_is_geodetic_):
      flattening = 1.0 - const.RPL_EARTH / const.REQ_EARTH
      equatorial_distance = np.sqrt(r_ecr_[0,:]**2 + r_ecr_[1,:]**2)

      beta = np.arctan( r_ecr_[2,:] / ((1.0 - flattening) * equatorial_distance) )
      latitude = np.arctan( (r_ecr_[2,:] + const.ECC_EARTH**2 * (1.0 - flattening) * const.REQ_EARTH * np.sin(beta_old)**3 / (1.0 - const.ECC_EARTH)) / (equatorial_distance - const.ECC_EARTH **2 * const.REQ_EARTH * np.cos(beta_old)**3) )

      beta = 0.0
      latitude = 0.0

      counter = 0

      while (np.fabs(latitude - latitude_old) > 1.0e-6 and counter < 25 ) :

         beta_old = beta
         latitude_old = latitude

         beta = np.arctan( (1.0 - flattening) * np.sin(latitude_old) / np.cos(latitude_old) )
         latitude = np.arctan( (r_ecr_[2,:] + const.ECC_EARTH**2 * (1.0 - flattening) * const.REQ_EARTH * np.sin(beta)**3 / (1.0 - const.ECC_EARTH)) / (equatorial_distance - const.ECC_EARTH **2 * const.REQ_EARTH * np.cos(beta)**3) )

         counter = counter + 1

      N = const.REQ_EARTH / np.sqrt(1.0 - const.ECC_EARTH**2 * np.sin(latitude)**2)
      altitude = equatorial_distance * np.cos(latitude) + (r_ecr_[2,:] + const.ECC_EARTH * N * np.sin(latitude)) * np.sin(latitude) - N
   
   else:
'''