#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: coordinate_transforms.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transform utilities
"""

# Import statements
import numpy as np
from scipy.interpolate import interp1d
#from numpy import linalg as LA

from . import physics_constants as const

# 
def wb2vbMatrix(t_, tsigmaTable_):
   """Construct matrix to convert from the WB to the VB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    """

   sigma = interp1d(tsigmaTable_[0,:], tsigmaTable_[1,:], kind='linear', bounds_error='false', fill_value=0.0)
   sigma = sigma(t_)# - 90.0

   topRow = np.multiply([1.0,                       0.0,                        0.0], np.ones(t_.size))
   midRow = np.multiply([0.0, np.sin(np.deg2rad(sigma)),  np.cos(np.deg2rad(sigma))], np.ones(t_.size))
   botRow = np.multiply([0.0, np.cos(np.deg2rad(sigma)), -np.sin(np.deg2rad(sigma))], np.ones(t_.size))

   Twb2vb = np.stack([topRow, midRow, botRow], axis=1)

   return Twb2vb, sigma

# 
def vb2wbMatrix(t_, tsigmaTable_):
   """Construct matrix to convert from the VB to the WB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    """

   Twb2vb,sigma = wb2vbMatrix(t_, tsigmaTable_)

   if(2 == len(Twb2vb.shape)):
      Tvb2wb = np.transpose(np.stack(Twb2vb, axis=1), (1,0))
   else:
      Tvb2wb = np.transpose(np.stack(Twb2vb, axis=1), (1,0,2))

   return Tvb2wb,sigma

# 
def vb2wb(t_, tsigmaTable_, invec_vb_):
   """Convert from the VB to the WB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    invec_vb_ -- 3 x N vector in VB coordinates that will be converted to the WB coordinates
    """
   # 3 x 3 x N matrix with basis unit vectors as rows
   Tvb2wb,sigma = vb2wbMatrix(t_, tsigmaTable_)

   invec_wb = []
   if(3 == len(Tvb2wb.shape)):
      # 3 x 1 x N column vector
      invec_vb_= invec_vb_[:,np.newaxis,:]
      # Multiple across the first two dimensions (sum over j) so result is ik and then broadcast along N dimension
      invec_wb = np.einsum('ij...,jk...->i...', Tvb2wb, invec_vb_)
   else:
      invec_wb = Tvb2wb.dot(invec_vb_)

   # 3 x N vector in wind basis
   return invec_wb, sigma

# 
def wb2vb(t_, tsigmaTable_, invec_wb_):
   """Convert from the VB to the WB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    invec_wb_ -- 3 x N vector in WB coordinates that will be converted to the VB coordinates
    """
   # 3 x 3 x N matrix with basis unit vectors as rows
   Tvb2wb,sigma = vb2wbMatrix(t_, tsigmaTable_)

   invec_vb = []
   if(3 == len(Tvb2wb.shape)):
      # 3 x 1 x N column vector
      invec_wb_ = invec_wb_[:,np.newaxis,:]
      # Multiple across the first two dimensions (sum over j) so result is ik and then broadcast along N dimension
      invec_vb = np.einsum('ij...,jk...->i...', Tvb2wb, invec_wb_)
   else:
      invec_vb = Tvb2wb.dot(invec_wb_)

   # 3 x N vector in velocity basis
   return invec_vb, sigma

# 
def vb2ecrMatrix(r_ecr_, v_ecr_):
   """Construct matrix to convert from the VB to the ECR basis.

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    vhat_ecr_ -- 3 x N velocity vector in ECR coordinates
    """
   rHat_ecr_ = r_ecr_ / np.linalg.vector_norm(r_ecr_,axis=0)
   vHat_ecr_ = v_ecr_ / np.linalg.vector_norm(v_ecr_,axis=0)

   xvbHat_ecr = vHat_ecr_
   yvbHat_ecr = np.cross(rHat_ecr_,xvbHat_ecr,axis=0)
   zvbHat_ecr = np.cross(xvbHat_ecr,yvbHat_ecr,axis=0)

   # 3 x 3 x N matrix with basis unit vectors as rows
   Tvb2ecr = np.stack([xvbHat_ecr,yvbHat_ecr,zvbHat_ecr], axis=1)

   return Tvb2ecr

#
def vb2ecr(r_ecr_, v_ecr_, invec_vb):
   """Convert from the VB to the ECR basis.

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    vhat_ecr_ -- 3 x N velocity vector in ECR coordinates
    invec_vb -- 3 x N vector in VB coordinates
    """
   # 3 x 3 x N matrix with basis unit vectors as rows
   Tvb2ecr = vb2ecrMatrix(r_ecr_, v_ecr_)

   invec_ecr = []
   if(3 == len(Tvb2ecr.shape)):
      # 3 x 1 x N column vector
      invec_vb = invec_vb[:,np.newaxis,:]
      # Multiple across the first two dimensions (sum over j) so result is ik and then broadcast along N dimension
      invec_ecr = np.einsum('ij...,jk...->i...', Tvb2ecr, invec_vb)
   else:
      invec_ecr = Tvb2ecr.dot(invec_vb)

   # 3 x N vector in ECR basis
   return invec_ecr

# 
def ecr2vbMatrix(r_ecr_, v_ecr_):
   """Convert from the VB to the ECR basis.

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    vhat_ecr_ -- 3 x N velocity vector in ECR coordinates
    """
   
   # 3 x 3 x N matrix with basis unit vectors as rows
   Tvb2ecr = vb2ecrMatrix(r_ecr_, v_ecr_)
   Tecr2vb = []
   if(2 == len(Tvb2ecr.shape)):
      Tecr2vb = np.transpose(np.stack(Tvb2ecr, axis=1), (1,0))
   else:
      Tecr2vb = np.transpose(np.stack(Tvb2ecr, axis=1), (1,0,2))

   return Tecr2vb

def ecr2vb(r_ecr_, v_ecr_, invec_ecr_):
   """Convert from the ECR to the VB basis.

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    vhat_ecr_ -- 3 x N velocity vector in ECR coordinates
    invec_vb -- 3 x N vector in VB coordinates
    """ 
   # 3 x 3 x N matrix with basis unit vectors as rows
   Tecr2vb = ecr2vbMatrix(r_ecr_, v_ecr_)

   invec_vb = []
   if(3 == len(Tecr2vb.shape)):
      # 3 x 1 x N column vector
      invec_ecr = invec_ecr[:,np.newaxis,:]
      # Multiple across the first two dimensions (sum over j) so result is ik and then broadcast along N dimension
      invec_vb = np.einsum('ij...,jk...->i...', Tecr2vb, invec_ecr_)
   else:
      invec_vb = Tecr2vb.dot(invec_ecr_)

   # 3 x N vector in VB basis
   return invec_vb

# 
def ecr2lla(r_ecr_, bool_is_geodetic_=False):
   """Convert position from from ECR coordinates to LLA

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    """
   if(1 == len(r_ecr_.shape)):
      r_ecr_ = r_ecr_[:,np.newaxis]

   longitude = np.arctan2(r_ecr_[1,:],r_ecr_[0,:]) # (y,x)
   
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
      latitude = np.arctan2(r_ecr_[2,:],np.linalg.vector_norm(r_ecr_[0:1,:],axis=0)) # (y,x)
      altitude = np.linalg.vector_norm(r_ecr_,axis=0) - const.R_EARTH

   return np.array([latitude, longitude, altitude])