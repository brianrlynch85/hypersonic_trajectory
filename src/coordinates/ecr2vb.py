#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: ecr2vb.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transforms from the ECR to velocity basis
"""

# Import statements
import numpy as np

from . import vb2ecr as m

# 
def ecr2vbMatrix(r_ecr_, v_ecr_):
   """Convert from the VB to the ECR basis.

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    vhat_ecr_ -- 3 x N velocity vector in ECR coordinates
    """
   
   # 3 x 3 x N matrix with basis unit vectors as rows
   Tvb2ecr = m.vb2ecrMatrix(r_ecr_, v_ecr_)
   Tecr2vb = []
   if(2 == len(Tvb2ecr.shape)):
      Tecr2vb = np.inv(Tvb2ecr)
   else:
      #Tecr2vb = np.transpose(Tvb2ecr, (1,0,2))
      Tecr2vb = np.linalg.inv(np.transpose(Tvb2ecr, (2,0,1)))
      Tecr2vb = np.transpose(Tecr2vb, (1,2,0))

   return Tecr2vb

#
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
      invec_ecr_ = invec_ecr_[:,np.newaxis,:]
      # Multiple across the first two dimensions (sum over j) so result is ik and then broadcast along N dimension
      invec_vb = np.einsum('ij...,jk...->i...', Tecr2vb, invec_ecr_)
   else:
      invec_vb = Tecr2vb.dot(invec_ecr_)

   # 3 x N vector in VB basis
   return invec_vb