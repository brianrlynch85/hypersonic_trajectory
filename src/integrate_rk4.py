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
import src.coordinate_transforms as coord
import src.forces as force

#
def integrate_rk4(r_ecr_, v_ecr_, sigmaTable_, t_, dt_):

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

