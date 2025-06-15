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

import pyatmos as atmo

#
def mass_density(r_ecr_):
   r"""Compute the atmospheric mass density

   Use the atmospheric mass density using the position vector in ECR coordiantes,
   assuming a spherical Earth, and a simple exponential decay model.

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates

   Returns
   -------
   Fcr : array_like(dtype=float, ndim=2)
      1 x N Armospheric mass density

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \rho (|\textbf{r}|) = ...

   References
   -----
   None

   """

   #rn = coord.ecr2lla(r_ecr_)[-1,:]
   rn = np.linalg.vector_norm(r_ecr_, axis=0) - const.R_EARTH

   #print(rn / 1000.0, atmo.coesa76(rn/1000.0).rho, const.rho0 * np.exp(-rn / const.H0))
   return const.rho0 * np.exp(-rn / const.H0)

# 
def gravitational_force(r_ecr_):
   r"""Compute the gravitational force

   Use the position in ECR coordinates to compute the Gravitational force

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates

   Returns
   -------
   Fcr : array_like(dtype=float, ndim=2)
      3 x N Gravitational force vector in ECR coordinates

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \textbf{F}_{g} = ...

   References
   -----
   None

   """

   rn = np.linalg.vector_norm(r_ecr_, axis=0)
   rHat_ecr_ = r_ecr_ / rn

   Fg = -const.GRAV_CONST * const.M_EARTH * const.M_VEHICLE / np.square(rn) * rHat_ecr_

   return Fg

# 
def coriolis_force(v_ecr_):
   r"""Compute the Coriolis force

   Use the velocity in ECR coordinates to compute the Coriolis force

   Parameters
   ----------
   v_ecr_ : array_like(dtype=float, ndim=2)
      3 x N velocity vector in ECR coordinates

   Returns
   -------
   Fcr : array_like(dtype=float, ndim=2)
      3 x N Coriolis force vector in ECR coordinates

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \textbf{F}_{cr} = ...

   References
   -----
   None

   """
   
   omega = const.W_EARTH
   if(1 != len(v_ecr_.shape)):
      omega = np.tile(omega[:,np.newaxis], v_ecr_.shape[1])

   Fcr = -2.0 * const.M_VEHICLE * np.cross(omega, v_ecr_, axis=0)

   return Fcr

#
def centrifugal_force(r_ecr_):
   r"""Compute the centrifugal force

   Use the position in ECR coordinates to compute the centrifugal force

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates

   Returns
   -------
   Fct : array_like(dtype=float, ndim=2)
      3 x N centrifugal force vector in ECR coordinates

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \textbf{F}_{ct} = ...

   References
   -----
   None

   """
   
   omega = const.W_EARTH
   if(1 != len(r_ecr_.shape)):
      omega = np.tile(omega[:,np.newaxis], r_ecr_.shape[1])

   Fct = -const.M_VEHICLE * np.cross(omega, np.cross(omega, r_ecr_, axis=0), axis=0)

   return Fct

#
def aero_force(r_ecr_, v_ecr_, sigmaTable_, t_):
   r"""Compute the aerodynamic force

   Use the position and velocity in ECR coordinates, a table of time versus
   bank angle, and current time to compute the aerodynamic force of a lifting
   vehicle

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates
   v_ecr_ : array_like(dtype=float, ndim=2)
      3 x N velocity vector in ECR coordinates
   sigmaTable_  : array_like(dtype=float, ndim=2)
      2 x M table of time and bank angles to be interpolated from
   t_ : array_like(dtype=float, ndim=1)
      N element time vector

   Returns
   -------
   Fa : array_like(dtype=float, ndim=2)
      3 x N aerodynamic force vector in ECR coordinates

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \textbf{F}_a = \frac{1}{2} \rho (|textbf{r}|) S v^2 \textbf{C}_{ecr}

   References
   -----
   None

   """

   C_w = const.CW_VEHICLE
   if(1 != len(r_ecr_.shape)):
      C_w = np.tile(C_w[:,np.newaxis],r_ecr_.shape[1])

   C_ecr,_ = coord.wind2ecr(t_, sigmaTable_, r_ecr_, v_ecr_, C_w)
   
   vn = np.linalg.vector_norm(v_ecr_, axis=0)
   Fa = -0.5 * mass_density(r_ecr_) * const.S_VEHICLE * np.square(vn) * C_ecr

   return Fa

#
def net_force(r_ecr_, v_ecr_, sigmaTable_, t_):
   r"""Compute the net force

   Compute the sum of the gravitational, Coriolis, centrifugal, and aerodynamic forces.

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates
   v_ecr_ : array_like(dtype=float, ndim=2)
      3 x N velocity vector in ECR coordinates
   sigmaTable_  : array_like(dtype=float, ndim=2)
      2 x M table of time and bank angles to be interpolated from
   t_ : array_like(dtype=float, ndim=1)
      N element time vector

   Returns
   -------
   Fn : array_like(dtype=float, ndim=2)
      3 x N Net force vector in ECR coordinates

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \textbf{F}_{net} = \textbf{F}_{g} + \textbf{F}_{cr} + \textbf{F}_{ct} + \textbf{F}_{a}

   References
   -----
   None

   """

   return gravitational_force(r_ecr_) + coriolis_force(v_ecr_) + centrifugal_force(r_ecr_) + aero_force(r_ecr_, v_ecr_, sigmaTable_, t_)

#
def net_acceleration(r_ecr_, v_ecr_, sigmaTable_, t_):
   r"""Compute the net acceleration

   Compute the sum of the gravitational, Coriolis, centrifugal, and aerodynamic accelerations.

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates
   v_ecr_ : array_like(dtype=float, ndim=2)
      3 x N velocity vector in ECR coordinates
   sigmaTable_  : array_like(dtype=float, ndim=2)
      2 x M table of time and bank angles to be interpolated from
   t_ : array_like(dtype=float, ndim=1)
      N element time vector

   Returns
   -------
   an : array_like(dtype=float, ndim=2)
      3 x N Net acceleration vector in ECR coordinates

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: \textbf{a}_{net} = \textbf{F}_{net} / m

   References
   -----
   None

   """
      
   return net_force(r_ecr_, v_ecr_, sigmaTable_, t_) / const.M_VEHICLE

