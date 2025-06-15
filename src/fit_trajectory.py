#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: fit_trajectory.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script uses a bank angle table to curve fit a trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
import scipy.interpolate as itp

import src.coordinate_transforms as coord
import src.physics_constants as const
import src.integrate_rk4 as slv
import src.forces as frc
import src.utils as utl

#
def residual(bank_angles_, r_ecr_, v_ecr_, to_, tf_, Nt_, control_point_times, time_altitude_data_, altitude_err_ = 1):

   bank_angle_table = np.nan * np.ones([len(bank_angles_), len(bank_angles_)])
   bank_angle_table[0,:] = control_point_times
   bank_angle_table[1,:] = bank_angles_

   r_ecr, v_ecr, time = slv.solvesystem(r_ecr_, v_ecr_, bank_angle_table, to_, tf_, Nt_)

   # Need to interpolate the solutions to match (time,altitude) values between data & sim
   # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splrep.html

   altitude_simulated = coord.ecr2lla(r_ecr)[2,:] / 1000.0 # km
   
   spline_params = itp.splrep(time, altitude_simulated) # Get spline parameters

   altitude_interpolated = itp.splev(time_altitude_data_[0,:], spline_params)

   # Calculate sum of differences (try scaling to improve convergence if needed)
   # This gets squared internally in the fitting routine
   ss = np.divide(np.subtract(altitude_interpolated, time_altitude_data_[1,:]), altitude_err_)

   return ss

#
def fit_trajectory(r_ecr_, v_ecr_, to_, tf_, Nt_, time_altitude_data_, altitude_err_):

   # Smooth the sampled trajectory data with splines
   time_interpolated = np.linspace(to_, tf_, Nt_)
   spline_params = itp.splrep(time_altitude_data_[0,:], time_altitude_data_[1,:])
   altitude_interpolated = itp.splev(time_interpolated, spline_params)

   # Get inflection and extrema of the smoothed trajectory. These are approximated as psuedo 'control' points.
   min1_indices, max1_indices = utl.get_extrema_indices(altitude_interpolated)
   min2_indices, max2_indices = utl.get_inflection_indices(altitude_interpolated)

   # Set initial guess for times and bank angles
   control_point_indices = np.sort([np.hstack((0, min1_indices, max1_indices, min2_indices, max2_indices))])
   control_point_times = np.array(time_interpolated[control_point_indices]).squeeze()
   control_point_times = np.append(control_point_times, tf_)
   utl.check_for_duplicates(control_point_times)

   bank_angle_guess = np.array(45.0 * np.ones(control_point_times.shape))
   bank_angle_guess[0] = 0.0
   bank_angle_guess[-1] = 0.0

   result = opt.least_squares(residual,
                          (bank_angle_guess),
                          args=(r_ecr_, v_ecr_, to_, tf_, Nt_, control_point_times, time_altitude_data_, altitude_err_),
                          bounds = (-90.0,90.0),
                          ftol=1.0e-3,
                          xtol=1.0e-4,
                          method='trf',
                          loss='soft_l1',
                          max_nfev=500)
   
   bank_angles = result.x
   bank_angle_table = np.nan * np.ones([len(bank_angles), len(bank_angles)])
   bank_angle_table[0,:] = control_point_times
   bank_angle_table[1,:] = bank_angles
 
   r_ecr, v_ecr, time = slv.solvesystem(r_ecr_, v_ecr_, bank_angle_table, to_, tf_, Nt_)

   #To-do
   #http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es

   return  r_ecr, v_ecr, time, bank_angle_table