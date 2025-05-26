#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: vb2ecr.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script contains coordinate transforms from the velocity to ECR basis
"""

# Import statements
import numpy as np

# 
def vb2ecrMatrix(r_ecr_, v_ecr_):
   """Construct matrix to convert from the VB to the ECR basis.

    Keyword arguments:
    rhat_ecr_ -- 3 x N position vector in ECR coordinates
    vhat_ecr_ -- 3 x N velocity vector in ECR coordinates
    """
   rHat_ecr_ = r_ecr_ / np.linalg.vector_norm(r_ecr_,axis=0)
   vHat_ecr_ = v_ecr_ / np.linalg.vector_norm(v_ecr_,axis=0)

   yvbHat_ecr = vHat_ecr_
   xvbHat_ecr = np.cross(rHat_ecr_,yvbHat_ecr,axis=0)
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