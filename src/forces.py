#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: forces.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains forces for predicting motion in a non-inertial reference frame.
"""

# Import statements
import numpy as np
#from numpy import linalg as LA

from . import physics_constants as const
import src.coordinate_transforms as coord

#
def mass_density(r_ecr_):
   """Compute the atmospheric mass density assuming simple exponential model and spherical Earth

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    """

   #rn = coord.ecr2lla(r_ecr_)[-1,:]
   rn = np.linalg.vector_norm(r_ecr_,axis=0) - const.R_EARTH
   return const.rho0 * np.exp(-rn / const.H0)

# 
def gravitational_force(r_ecr_):
   """Compute the gravitational force.

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    """

   rn = np.linalg.vector_norm(r_ecr_,axis=0)
   rHat_ecr_ = r_ecr_ / rn

   Fg = -const.GRAV_CONST * const.M_EARTH * const.M_VEHICLE / np.square(rn) * rHat_ecr_

   return Fg

# 
def coriolis_force(v_ecr_):
   """Compute the Coriolis force.

    Keyword arguments:
    v_ecr_ -- 3 x N velocity vector in ECR coordinates
    """
   
   omega = np.multiply([0.0, 0.0, const.W_EARTH], np.ones(v_ecr_.shape))
   Fcr = -2.0 * const.M_VEHICLE * np.cross(omega, v_ecr_, axis=0)

   return Fcr


def centrifugal_force(r_ecr_):
   """Compute the centrifugal force.

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    """
   
   omega = np.multiply([0.0, 0.0, const.W_EARTH], np.ones(r_ecr_.shape))
   Fcr = -const.M_VEHICLE * np.cross(omega, np.cross(omega, r_ecr_, axis=0), axis=0)

   return Fcr

#
def aero_force(r_ecr_, v_ecr_, sigmaTable_, t_):
   """Compute the aerodynamic force.

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    v_ecr_ -- 3 x N velocity vector in ECR coordinates
    sigmaTable_ -- 3 x N bank angle table
    t_ -- N element time vector
    """
   
   C_w = np.multiply(const.CW_VEHICLE, np.ones(t_.size))
   C_vb,__ = coord.wb2vb(t_, sigmaTable_, C_w)
   C_ecr = coord.vb2ecr(r_ecr_, v_ecr_, C_vb)
   
   vn = np.linalg.vector_norm(v_ecr_, axis=0)
   Fd = -0.5 * mass_density(r_ecr_) * const.S_VEHICLE * np.square(vn) * C_ecr

   return Fd

#
def net_force(r_ecr_, v_ecr_, sigmaTable_, t_):
   """Compute the net force

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    v_ecr_ -- 3 x N velocity vector in ECR coordinates
    sigmaTable_ -- 3 x N bank angle table
    t_ -- N element time vector
    """
   
   return gravitational_force(r_ecr_) + coriolis_force(v_ecr_) + centrifugal_force(r_ecr_) + aero_force(r_ecr_, v_ecr_, sigmaTable_, t_)

#
def net_acceleration(r_ecr_, v_ecr_, sigmaTable_, t_):
   """Compute the net acceleration.

    Keyword arguments:
    r_ecr_ -- 3 x N position vector in ECR coordinates
    v_ecr_ -- 3 x N velocity vector in ECR coordinates
    sigmaTable_ -- 3 x N bank angle table
    t_ -- N element time vector
    """
      
   return net_force(r_ecr_, v_ecr_, sigmaTable_, t_) / const.M_VEHICLE

