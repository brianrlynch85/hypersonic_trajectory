#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: ecr2lla.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transforms from latitude, longitude, altitude to ECR
"""

import numpy as np

from src import physics_constants as const

# 
def lla2ecr(lla_, bool_is_geodetic_=False):
   """Convert position from from LLA to ECR coordinates

    Keyword arguments:
    lla_ -- 3 x N position vector in LLA coordinates
    """
   
   if(1 == len(lla_.shape)):
      lla_ = lla_[:,np.newaxis]

   r = lla_[2,:] + const.R_EARTH

   x_ecr = r * np.array([np.cos(np.deg2rad(lla_[1,:])) * np.cos(np.deg2rad(lla_[0,:]))])
   y_ecr = r * np.cos(np.deg2rad(lla_[1,:])) * np.sin(np.deg2rad(lla_[0,:]))
   z_ecr = r * np.sin(np.deg2rad(lla_[1,:]))

   return np.array([x_ecr, y_ecr, z_ecr])