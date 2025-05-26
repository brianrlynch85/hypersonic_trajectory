#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: wb2vb.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transforms from the wind to velocity basis
"""

# Import statements
import numpy as np
from scipy.interpolate import interp1d

from . import vb2wb as m

# 
def wb2vbMatrix(t_, tsigmaTable_):
   """Construct matrix to convert from the VB to the WB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    """

   Twb2vb,sigma = m.vb2wbMatrix(t_, tsigmaTable_)

   if(2 == len(Twb2vb.shape)):
      Tvb2wb = np.transpose(Twb2vb, (1,0))
   else:
      Tvb2wb = np.transpose(Twb2vb, (1,0,2))

   return Tvb2wb,sigma

# 
def wb2vb(t_, tsigmaTable_, invec_wb_):
   """Convert from the WB to the VB coordinates.

    Keyword arguments:
    t_ -- N element vector of time points to interpolate at 
    tsigmaTable_ -- 2 x N vector of time and bank angle values to interpolate from
    invec_wb_ -- 3 x N vector in WB coordinates that will be converted to the VB coordinates
    """
   # 3 x 3 x N matrix with basis unit vectors as rows
   Twb2vb,sigma = wb2vbMatrix(t_, tsigmaTable_)

   invec_vb = []
   if(3 == len(Twb2vb.shape)):
      # 3 x 1 x N column vector
      invec_wb_ = invec_wb_[:,np.newaxis,:]
      # Multiple across the first two dimensions (sum over j) so result is ik and then broadcast along N dimension
      invec_vb = np.einsum('ij...,jk...->i...', Twb2vb, invec_wb_)
   else:
      invec_vb = Twb2vb.dot(invec_wb_)

   # 3 x N vector in velocity basis
   return invec_vb, sigma