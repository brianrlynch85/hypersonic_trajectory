#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: fit_trajectory.py
Author: Brian Lynch
Date: 2024-08-12
Version: 1.0
Description: This script uses a bank angle table to curve fit a trajectory
"""

import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
from scipy.optimize import fmin,leastsq
from scipy.stats import chisquare
from scipy.interpolate import splev, splrep
from matplotlib.mlab import griddata
from matplotlib import ticker

import src.integrate_rk4 as slv



#
def residual(params, init0, t, xdata, ydata, xdataErr, ydataErr, Bz):

    xT,vxT,yT,vyT,tT = slv.solvesystem(params, Bz, init0, t)

    # Need to interpolate the solutions to match (x,y) from data
    # Decided to use cublic splines
    #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splrep.html
    #interX = splrep(np.flipud(yT), np.flipud(xT)) # Get spline params

    # Calculate the new x-locations from the spline knots
    #xfitT = splev(ydata, interX)

    matchInd = []; i = 0
    for yvalue in ydata:
       matchInd.append(np.abs(np.subtract(yT,yvalue)).argmin())

    xfitT = xT[matchInd]
    yfitT = yT[matchInd]

    #Calculate sum of differences (try scaling to improve convergence)
    #ss = np.sqrt(np.divide(np.subtract(xdata,xfitT),xdataErr)**2 + np.divide(np.subtract(ydata,yfitT),ydataErr)**2)
    # This gets squared internally in the fitting routine
    ss = np.divide(np.subtract(xdata,xfitT),xdataErr)
    return ss

###############################################################################
###############################################################################
def ODE_Fit(filename,data1,Bz,ShowGraphs=True):

   ###############################################################################
   # Setup the initial conditions and guess parameters
   t = np.arange(0.0,1.0,1.0/(50000))
   #x = xo; y = yo; vx = vxo; vyo = -g/GAMMAC

   init0 = (xo,vxo,yo,vyo)
   params0 = [-330.0,2500.0]

   ###############################################################################
   # Start solving the system

   # Guess at the parameters and see results (checking 1st iteration)
   #x,vx,y,vy,tt = solvesystem(params0, init0, t)

   # Convert file arrrays into real coordinates
   ydata    = -(reso * np.asarray(data1[:,2],dtype=np.float64) - np.asarray(ycenImage,dtype=np.float64)) # Offset y and * (-1)
   xdata    = reso*np.asarray(data1[:,0],dtype=np.float64)
   xdataErr = reso*np.asarray(data1[:,1],dtype=np.float64)
   ydataErr = reso*np.asarray(data1[:,3],dtype=np.float64)

    '''
    ax = plt.gca()
    ax.errorbar(100.0*10*np.array(xdata),100.0*np.array(ydata) ,xerr=xdataErr, fmt='bo')
    plt.show(); plt.clf()
    '''

    # Now solve the system and calculate the appropriate parameters
    #answ=fmin(chi_squared, (params0), args=(init0, t, xdata, ydata,xdataErr), xtol=1.0e-10, \
    #                                      ftol=1.0e-10, full_output=1, maxiter=5000)
   Fdust,fitCov,fitInfoDict,sMsg,s = leastsq(residual,(params0),
                                args=(init0,t,xdata,ydata,xdataErr,ydataErr,Bz),
                                                      ftol=1.0e-10,xtol=1.0e-10,
                                                      full_output=1,maxfev=5000)

   xfit, vxfit, yfit, vyfit, tfit = solvesystem(Fdust,Bz,init0,t)
   xguess, vxguess, yguess, vyguess, tguess = solvesystem(params0,Bz,init0,t)

   plt.plot(1000*np.array(xguess),1000.0*np.array(yguess),'r--',)
   plt.plot(1000*np.array(xfit),1000.0*np.array(yfit),'b--')
   ax = plt.gca()
   ax.errorbar(1000.0*np.array(xdata),1000.0*np.array(ydata) ,xerr=xdataErr, fmt='bo')
   plt.axis([-1.27,1.27,-1000.0*YMAX,1000.0*yo])
   ax.grid(True, which='both')
   plt.show(); plt.clf()

   #http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
   #ZdustError = 0.0#np.sqrt(np.multiply((fitInfoDict['fvec']**2).sum() / (len(xdata) - 1),fitCov))

   return  -Zdust,ZdustError,MHallparam,MHallparamOML

###############################################################################
###############################################################################
def main():

    B = [-0.512,-0.768,-1.024,-1.248,-1.504,-1.760,-2.016,-2.240]
    P = 5.0

    Bvalues = []
    Zvalues = []
    ZstdError = []
    MHall = []
    MHallOML = []

    print('ALL OUTPUT TEXT FILES SHOULD CONTAIN STANDARDS ERRORS NOT COVARIANCE ENTRIES')

    for bitem in B:
        print('INPUT PARAMETERS: B (T) = ', bitem,' , P (mTorr) = ', P)
        Bstr = "{:4.3f}".format(-bitem); Pstr = str(P)
        file_pref = 'XY_IO/B' + Bstr.replace(".","p") + 'P' + Pstr.replace(".","p")

        print('file prefix: ', file_pref)
        filename = file_pref + 'DeflectionDataPixels.txt'
        data1 = np.loadtxt(filename,skiprows=0)
        print('INPUT FILENAME: ',filename)

        Ztemp,ZstdTemp,MHallTemp,MHallOMLTemp = ODE_Fit(filename,data1,bitem,True)
        Zvalues.append(Ztemp); ZstdError.append(ZstdTemp); Bvalues.append(-bitem)
        MHall.append(MHallTemp); MHallOML.append(MHallOMLTemp)

    print('Done ALL ODE curve fits')


    ############################################################################
    # Plot the dust charge as function of magnetic field
    ax = plt.gca()
    ax.errorbar(Bvalues,Zvalues,yerr=np.array(ZstdError),fmt='ko')
    ax.set_ylabel(r'Dust Charge $Z_{d}$ (e)')
    ax.set_xlabel('Magnetic Field B (T)')
    ax.set_title('Dust Charge Summary: ' + r'$v_{o,y}$ = ' + str(round(vyo,2)) + ' m/s')
    plt.axis([0.0,3.0,0.0,100.0])

    ax.grid(True, which='both')
    #ax.set_title(r'Charge Measurement Summary: $a_{d}$ = 0.25 $\mu$m')
    plt.legend(loc='upper right')

    # Now add a top axis for the ion Larmor radius
    new_xtick_locations = np.array([0.5,1.0,1.5,2.0,2.5])
    def xtick_function(X):
        V = 1000000*(m_e * vt_e / X / echarge)
        print(V)
        return ["%.1f" % z for z in V]
    #ax2 = ax.twiny()
    #ax2.set_xlim(ax.get_xlim())
    #ax2.set_xticks(new_xtick_locations)
    #ax2.set_xticklabels(xtick_function(new_xtick_locations))
    #ax2.set_xlabel(r'2.0 eV Electron Larmor Radius $r_{L,e}$ ($\mu$m)')

    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig(filename[:-4] +'2parvyo'+str(-vyo).replace('-','').replace('.','p')+ '2pargxbZdvsB.png',bbox_inches='tight')
    #plt.savefig(filename[:-4] + 'gxbZdvsB.eps',bbox_inches='tight')
    plt.show(); plt.clf()

    ############################################################################
    # Plot the dust charge as function of larmor radii
    ax = plt.gca()

    reLarm = 1000000.0*np.divide(m_e*vt_e/echarge,Bvalues)

    ax.errorbar(reLarm,Zvalues,yerr=np.array(ZstdError),fmt='ko')
    ax.set_ylabel(r'Dust Charge $Z_{d}$ (e)')
    ax.set_xlabel(r'2.0 eV Electron Larmor Radius $r_{L,e}$ ($\mu$m)')
    plt.axis([0.0,12.25,0.0,100.0])
    ax.grid(True, which='both')

    # Now add a top axis for the ion Larmor radius
    new_xtick_locations = np.array([2.0,4.0,6.0,8.0,10.0])
    def xtick_function(X):
        V = m_n * vt_n / (m_e * vt_e / X / echarge) / echarge
        print(V)
        return ["%.1f" % z for z in V]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_xtick_locations)
    ax2.set_xticklabels(xtick_function(new_xtick_locations))
    ax2.set_xlabel(r'0.025 eV Ion Larmor Radius $r_{L,i}$ ($\mu$m)')

    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig(filename[:-4] +'2parvyo'+str(-vyo).replace('-','').replace('.','p')+ '2pargxbZdvsrLe.png',bbox_inches='tight')
    #plt.savefig(filename[:-4] + 'gxbZdvsrLe.eps',bbox_inches='tight')
    plt.show(); plt.clf()

    ############################################################################
    # Plot the modified Hall parameters for data and 30% OML
    ax = plt.gca()
    plt.plot(Bvalues,MHall,'ko')
    #ax.set_yscale('log')
    ax.set_xlabel(r'Magnetic Field B (T)')
    ax.set_ylabel(r'Modified Hall Parameter $H_{\mathrm{m,d}}$')
    plt.xlim(0.0,3.0)
    plt.ylim(0.00001,0.00125)

    ax.grid(True, which='both')
    ax.set_title(r'Dust Hall Parameter Summary: ' + r'$v_{o,y}$ = ' + str(round(vyo,2)) + ' m/s')
    plt.legend(loc='upper right')

    # Now add a right axis for the dust hall parameter
    new_ytick_locations = np.array([0.0002,0.0004,0.0006,0.0008,0.001])
    def ytick_function(X):
        V = X * 2.0 * math.pi
        print(V)
        return ["%.4f" % z for z in V]
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(new_ytick_locations)
    ax3.set_yticklabels(ytick_function(new_ytick_locations))
    ax3.set_ylabel(r'Hall Parameter $H_{\mathrm{d}}$')

    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams.update({'font.size': 16})
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    plt.savefig(filename[:-4] +'2parvyo'+str(-vyo).replace('-','').replace('.','p')+ '2pargxbHallvsB.png',bbox_inches='tight')
    #plt.savefig(filename[:-4] + 'gxbHallvsB.eps',bbox_inches='tight')
    plt.show(); plt.clf()

    #############################################################################
    # Dust B field and pressure range

    ax = plt.gca()
    xd,yd,Xd,Yd,Zd,Pd,Bd = DustHallMParam(r_d,rho_d,-1.0,3.0,330*echarge,-3.0,0.802)

    levels = [0.0001,0.001, 0.01, 0.1, 1.0]
    CS = ax.contour(Pd, Bd, Zd,levels,colors=['k','b','g','r','m'],linewidths=2.0,locator=ticker.LogLocator())
    plt.title(r'Silica Dust Modified Hall Parameter $H_{\mathrm{m,d}}$')

    # Label the plot axes and the contour
    ax.set_xlabel(r'Neutral Pressure $p_{\mathrm{n}}$ (mTorr)')
    ax.set_ylabel(r'Magnetic Field B (T)')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.67,0.0047,'0.0001',fontweight='bold',fontsize='13',rotation=38,color='k')
    ax.text(0.67,0.0407,'0.001',fontweight='bold',fontsize='13',rotation=38,color='b')
    ax.text(0.67,0.36,'0.01',fontweight='bold',fontsize='13',rotation=38,color='g')
    ax.text(0.67,2.999,'0.1',fontweight='bold',fontsize='13',rotation=38,color='r')
    ax.text(0.12,4.5,'1',fontweight='bold',fontsize='13',rotation=38,color='m')
    ax.text(65,0.0065,r'$k_bT_n$ = 0.025 eV',fontweight='bold',fontsize='15',color='k')
    ax.text(98,0.0030,r'$Z_d$ = 330 e',fontweight='bold',fontsize='15',color='k')
    ax.text(98,0.0045,r'$a_d$ = 0.25 $\mu$m',fontweight='bold',fontsize='15',color='k')

    ax.grid(which='both')
    # Save the figure to file
    fig = plt.gcf()
    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams.update({'font.size': 16})
    plt.tight_layout()
    #fig.savefig('DustHallParameterContoursZ330.eps',bbox_inches='tight')
    fig.savefig('2parDustHallParameterContoursZ330.png',bbox_inches='tight')
    plt.show(); plt.clf()


    return

if __name__=='__main__':
   main()
