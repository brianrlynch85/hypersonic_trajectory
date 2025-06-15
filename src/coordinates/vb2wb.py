#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: vb2wb.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transforms from the velocity to wind basis
"""

# Import statements
import numpy as np
from scipy.interpolate import interp1d

# 
def vb2wbMatrix(t_, tsigmaTable_, kind_='linear'):
   """Construct matrix to convert from the WB to the VB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    """

   sigma = interp1d(tsigmaTable_[0,:], tsigmaTable_[1,:], kind=kind_, bounds_error='false', fill_value=0.0)
   sigma = sigma(t_)# - 90.0

   topRow = np.multiply([np.sin(np.deg2rad(sigma)), np.zeros(sigma.shape), -np.cos(np.deg2rad(sigma))], np.ones(t_.size))
   midRow = np.multiply([np.zeros(sigma.shape)    , np.ones(sigma.shape) ,  np.zeros(sigma.shape)                      ], np.ones(t_.size))
   botRow = np.multiply([np.cos(np.deg2rad(sigma)), np.zeros(sigma.shape),  np.sin(np.deg2rad(sigma))], np.ones(t_.size))

   Twb2vb = np.stack([topRow, midRow, botRow], axis=1)

   return Twb2vb, sigma

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
def deriv1bank_vb2wb(t_, tsigmaTable_, invec_vb_):
   return 1