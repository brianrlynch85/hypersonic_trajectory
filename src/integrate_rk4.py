#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: forces.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains a 4th order Runge Kutta integration of equations of motion
"""

# Import statements
import numpy as np
#from numpy import linalg as LA

from . import physics_constants as const

import src.forces as force

#
def integrate_rk4(r_ecr_, v_ecr_, sigmaTable_, t_, dt_):
   r"""Integrate forward a single timestep using a 4th order Runge Kutta

   Integrate forward a single timestep dt using the 4th order Runge kutta method

   Parameters
   ----------
   r_ecr_ : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates
   v_ecr_ : array_like(dtype=float, ndim=2)
      3 x N velocity vector in ECR coordinates
   sigmaTable_  : array_like(dtype=float, ndim=2)
      2 x M table of time and bank angles to be interpolated from
   t_ : scalar(dtype=float, ndim=0)
      Current time prior to integrating forward
   dt_ : scalar(dtype=float, ndim=0)
      timestep 

   Returns
   -------
   rsv : array_like(dtype=float, ndim=1)
      3 Element position vector in ECR coordinates
   vsv : array_like(dtype=float, ndim=1)
      3 Element velocity vector in ECR coordinates
   ts : scalar(dtype=float, ndim=0)
      New time after integrating forward by dt

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: ...

   References
   -----
   None

   """

   k1rdot = v_ecr_
   k1vdot = force.net_acceleration(r_ecr_, v_ecr_, sigmaTable_, t_)

   k2rdot = v_ecr_ + 0.5 * dt_ * k1vdot
   k2vdot = force.net_acceleration(r_ecr_ + 0.5 * dt_ * k1rdot, v_ecr_ + 0.5 * dt_ * k1vdot, sigmaTable_, t_ + 0.5 * dt_)

   k3rdot = v_ecr_ + 0.5 * dt_ * k2vdot
   k3vdot = force.net_acceleration(r_ecr_ + 0.5 * dt_ * k2rdot, v_ecr_ + 0.5 * dt_ * k2vdot, sigmaTable_, t_ + 0.5 * dt_)

   k4rdot = v_ecr_ + dt_ * k3vdot
   k4vdot = force.net_acceleration(r_ecr_ + dt_ * k3rdot, v_ecr_ + dt_ * k3vdot, sigmaTable_, t_ + dt_)

   rsv = r_ecr_ + dt_ * (k1rdot + 2.0 * k2rdot + 2.0 * k3rdot + k4rdot) / 6.0
   vsv = v_ecr_ + dt_ * (k1vdot + 2.0 * k2vdot + 2.0 * k3vdot + k4vdot) / 6.0
   ts = t_ + dt_

   return rsv, vsv, ts

#
def solvesystem(r0_ecr_, v0_ecr_, sigmaTable_, to_, tf_, Ntimepoints_):
   r"""Integrate the sytem forward in time from initial to final time

   Integrate forward all timesteps from initial time to the final time using
   the 4th order Runge kutta method

   Parameters
   ----------
   r0_ecr_ : array_like(dtype=float, ndim=1)
      3 element initial position vector in ECR
   v0_ecr_ : array_like(dtype=float, ndim=1)
      3 element initial velocity vector in ECR
   sigmaTable_  : array_like(dtype=float, ndim=2)
      2 x M table of time and bank angles to be interpolated from
   t0_ : scalar(dtype=float, ndim=0)
      Initial time
   tf_ : scalar(dtype=float, ndim=0)
      Final time
   Ntimepoints_ : scalar(dtype=float, ndim=0)
      Total # of timepoints to use when integrating

   Returns
   -------
   r_ecr : array_like(dtype=float, ndim=2)
      3 x N position vector in ECR coordinates
   v_ecr : array_like(dtype=float, ndim=2)
      3 x N velocity vector in ECR coordinates
   time : array_like(dtype=float, ndim=1)
      N element time vector

   Other Parameters
   ----------------
   None

   Raises
   ------
   None

   Notes
   -----
   .. math:: ...

   References
   -----
   None

   """
      
   time = np.linspace(to_, tf_, Ntimepoints_ + 1)

   r_ecr = np.empty([3, Ntimepoints_ + 1])
   v_ecr = np.empty([3, Ntimepoints_ + 1])

   r_ecr[:,0] = r0_ecr_
   v_ecr[:,0] = v0_ecr_

   for i in range(0, Ntimepoints_):
      dt = time[i+1] - time[i]
      r_ecr[:,i+1], v_ecr[:,i+1], _ = integrate_rk4(r_ecr[:,i], v_ecr[:,i], sigmaTable_, time[i], dt)

   return r_ecr, v_ecr, time

