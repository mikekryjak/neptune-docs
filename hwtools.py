import os, sys
import numpy as np
from datetime import datetime as dt
import xbout
import xarray as xr

class HWcase():
    
    
    def __init__(self):
        
        pass
    
    def load(self, path):
        squash(path)

        self.ds = xbout.load.open_boutdataset(
                    datapath = os.path.join(path,"BOUT.squash.nc"),
                    inputfilepath = os.path.join(path, "BOUT.inp"),
                    info = False,
                    cache = False,
                    keep_xboundaries=False,
                    keep_yboundaries=False,
                    )
        
    def set_params(self, 
        T0 = 50,            # Temperature [eV]
        n0 = 1e19,          # Density [m^-3]
        B = 0.5,            # Magnetic field [T]
        Z = 1,              # Ion charge
        lambda_n = 0.015,     # SOL radial density decay length
        ):
        
        self.ds.metadata["T0"] = T0
        self.ds.metadata["n0"] = n0
        self.ds.metadata["B"] = B
        self.ds.metadata["Z"] = Z
        self.ds.metadata["lambda_n"] = lambda_n
        
        
    def unnormalise(self):
        
        ds = self.ds
        m = ds.metadata
        
        ### Constants
        Mp = constants("Mp")
        Mi = constants("Mi")
        Me = 9.1093837e-31 # Electron mass [kg]
        qe = 1.60217662E-19 # electron charge [C] or [J ev^-1]
        e0 = 8.854187817e-12 # Vacuum permittivity [F m^-1]
        kb = 1.38064852e-23 # Boltzmann constant [J K^-1]
        
        
        B = m["B"]
        T0 = m["T0"]
        n0 = m["n0"]
        Z = m["Z"]
        lambda_n = m["lambda_n"]

        ### Normalisations
        omega_ci = qe * B / Mi   # ion gyrofrequency [s^-1]
        omega_ce = qe * B / Me   # electron gyrofrequency [s^-1]
        Cs0 = np.sqrt( (qe * T0) / Mi)   # Speed [m s^-1]
        rho_s0 = np.sqrt(Mi*T0/(qe*B))  # Distance, from hybrid Larmor radius [m]

        ds["n"] *= n0   # Density [m^-3]
        ds["phi"] *= T0   # Potential [V]
        ds["vort"] *= n0*qe   # Vorticity [???]

        ### Make X coordinate from dx (https://github.com/boutproject/xBOUT-examples/blob/master/hasegawa-wakatani/hasegawa-wakatani_example.ipynb)
        dx = ds["dx"].isel(x=0).values

        # Get rid of existing "x" coordinate, which is just the index values.
        ds = ds.drop("x")

        # Create a new coordinate, which is length in units of rho_s
        ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx[0])

        ### Unnormalise all spatial coordinates
        # Note Xarray won't let you do it inplace, hence this method
        ds = ds.assign_coords(x=ds["x"]*rho_s0)
        ds = ds.assign_coords(y=ds["y"]*rho_s0)
        ds = ds.assign_coords(z=ds["z"]*rho_s0)
        
        ## Unnormalise grid spacing
        ds["dx"] *= rho_s0
        ds["dy"] *= rho_s0
        ds["dz"] *= rho_s0

        ### Unnormalise time
        ds = ds.assign_coords(t=ds["t"]*(1/omega_ci))
        
        ### Add useful parameters to metadata
        ds.metadata["T0"] = T0
        ds.metadata["n0"] = n0
        ds.metadata["B"] = B
        ds.metadata["Z"] = Z
        ds.metadata["rho_s0"] = rho_s0
        
        
        ### Alpha and kappa calcs
            ## From Hermes-3 manual: 
        # https://hermes3.readthedocs.io/en/latest/components.html#collisions
        ## Originally from NRL formulary but converted to SI from CGS, hence different constants:
        # https://library.psfc.mit.edu/catalog/online_pubs/NRL_FORMULARY_19.pdf

        if T0 < 0.1 or n0 < 1e10:
            clog = 10   
        elif T0*Me/Mi < T0 and T0 < 10*Z**2:
            clog = 30 - 0.5*np.log(n0) + 1.5*np.log(T0)
        elif T0*Me/Mi < 10*Z**2 and 10*Z**2 <= T0:    # <--------- This one for T0=50, n0=1e19
            clog = 31 - 0.5*np.log(n0) + np.log(T0)
        elif T0 < T0*Me/Mi:
            clog = 23 - 0.5*np.log(n0) + 1.5*np.log(T0) - np.log(Z*Mi)
        else:
            raise Exception(f"T0 {T0} and n0 {n0} not in range of NRL formulary")
                

        # Collision time from Fitzpatrick https://farside.ph.utexas.edu/teaching/plasma/lectures1/node35.html
        T0_celsius = T0 * qe / kb
        T0_joules = T0 * qe

        eta = (np.pi * qe**2 * np.sqrt(Me)) / (4 * np.pi * e0)**2 * (T0 * qe)**(3/2) * clog
        eta_spitzer = 5.2e-5 * (Z*clog) / (T0**(3/2))  # T0 in eV
        
        Lpar = ds.isel(t=-1, x=0,z=0,y=0)["dy"].values * ds.metadata["ny"] * rho_s0
        k = 2*np.pi / Lpar    # Parallel wavenumber, should be y-direction representative wavenumber (2pi/lpar)
        # k = 1
        alpha = (T0 * 1**2) / (n0 * qe * eta_spitzer * omega_ci)
        alpha_2d = (T0 * k**2) / (n0 * qe * eta_spitzer * omega_ci)
        
        kappa = rho_s0 / lambda_n
        
        ds.metadata["alpha"] = alpha
        ds.metadata["alpha2d"] = alpha_2d
        ds.metadata["kappa"] = kappa
        ds.metadata["Lpar"] = Lpar
        
        
        self.ds = ds
        
        
        
        
        print(f"Domain extents:")
        print(f"X: {ds['x'].min().values*1e3:.1f} - {ds['x'].max().values*1e3:.1f} mm")
        print(f"Y: {ds['y'].min().values*1e3:.1f} - {ds['y'].max().values*1e3:.1f} mm")
        print(f"Z: {ds['z'].min().values*1e3:.1f} - {ds['z'].max().values*1e3:.1f} mm")
        print(f"\nTime simulated: {ds['t'].max().values:.1e} s")
        
    def calculate_energy(self):
        """
        Get integral of KE and PE in the domain as per Korsholm 1999 but in SI units
        """
        
        qe = constants("qe")
        ds = self.ds
        m = ds.metadata

        n_fl = ds["n"].values 
        n = n_fl + m["n0"]
        phi_fl = ds["phi"].values
        phi = phi_fl + m["T0"]

        vort = ds["vort"].values  # Vort has no floor

        dx = ds["dx"].values.flatten()[0]
        dy = ds["dy"].values.flatten()[0]
        dz = ds["dz"].values.flatten()[0]
        dv = dx * dy * dz

        gradPhi_all = np.gradient(phi, dx, dy, dz, axis = (1,2,3))   # List of gradients along x,y,z
        gradPhi = np.sum(gradPhi_all, axis = 0)   # Sum of gradients 
        gradPhiPar = gradPhi[1]
        gradPhiPerp = gradPhi - gradPhiPar

        # Kinetic energy is equivalent to 1/2 mv^2 where the velocity comes from ExB
        B = m["B"]
        E = -gradPhiPerp 
        v = (E * B)/B**2  # ExB/B^2. B is constant so this is easy

        KE = 0.5 * v**2 * m["n0"] * constants("Mi")    #  [J/m3] Using the background density as per the Bousinessq approximation
        KE = np.sum(KE,axis=(1,2,3)) * dv


        # Potential energy is the absolute of the density deviation times background temp and integrated over volume
        PE = n_fl**2 / m["n0"] * m["T0"] * constants("qe")    # [J/m3]
        PE = np.sum(PE,axis=(1,2,3)) * dv
        
        ds["PE"] = xr.DataArray(PE, dims = ["t"])
        ds["KE"] = xr.DataArray(KE, dims = ["t"])
        
        self.ds = ds
        
        
        
def squash(casepath, verbose = True, force = False):
    """
    Checks if squashed file exists. If it doesn't, or if it's older than the dmp files, 
    then it creates a new squash file.
    
    Inputs
    ------
    Casepath is the path to the case directory
    verbose gives you extra info prints
    force always recreates squash file
    """
    
    datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
    inputfilepath = os.path.join(casepath, "BOUT.inp")
    squashfilepath = os.path.join(casepath, "BOUT.squash.nc") # Squashoutput hardcoded to this filename

    recreate = True if force is True else False   # Override to recreate squashoutput
    squash_exists = False
    
    if verbose is True: print(f"- Looking for squash file")
        
    if "BOUT.squash.nc" in os.listdir(casepath):  # Squash file found?
        
        squash_exists = True
        
        squash_date = os.path.getmtime(squashfilepath)
        dmp_date = os.path.getmtime(os.path.join(casepath, "BOUT.dmp.0.nc"))
        
        squash_date_string = dt.strftime(dt.fromtimestamp(squash_date), r"%m/%d/%Y, %H:%M:%S")
        dmp_date_string = dt.strftime(dt.fromtimestamp(dmp_date), r"%m/%d/%Y, %H:%M:%S")
        
        if verbose is True: print(f"- Squash file found. squash date {squash_date_string}, dmp file date {dmp_date_string}") 
        
        if dmp_date > squash_date:   #Recreate if squashoutput is too old
            recreate = True
            print(f"- dmp files are newer than the squash file! Recreating...") 
            
    else:
        if verbose is True: print(f"- Squashoutput file not found, creating...")
        recreate = True
        

    if recreate is True:
        
        if squash_exists is True:  # Squashoutput will not overwrite, so we delete the file first
            os.remove(squashfilepath)
            
        squashoutput(
            datadir = casepath,
            outputname = squashfilepath,
            xguards = True,   # Take all xguards
            yguards = "include_upper",  # Take all yguards (yes, confusing name)
            parallel = False,   # Seems broken atm
            quiet = verbose
        )
        
        if verbose is True: print(f"- Done")
            



def unnormalise_hw3d(ds, T0, n0, B, Z):
    
    
    

    ### Constants
    Mp = 1.6726219e-27 # Proton mass [kg]
    Mi = Mp*2  # Ion mass of deuterium [kg]
    Me = 9.1093837e-31 # Electron mass [kg]
    qe = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    e0 = 8.854187817e-12 # Vacuum permittivity [F m^-1]
    kb = 1.38064852e-23 # Boltzmann constant [J K^-1]

    ### Normalisations
    omega_ci = qe * B / Mi   # ion gyrofrequency [s^-1]
    omega_ce = qe * B / Me   # electron gyrofrequency [s^-1]
    Cs0 = np.sqrt( (qe * T0) / Mi)   # Speed [m s^-1]
    rho_s0 = np.sqrt(Mi*T0/(qe*B))  # Distance, from hybrid Larmor radius [m]

    ds["n"] *= n0   # Density [m^-3]
    ds["phi"] *= T0   # Potential [V]
    ds["vort"] *= n0*qe   # Vorticity [???]

    ### Make X coordinate from dx (https://github.com/boutproject/xBOUT-examples/blob/master/hasegawa-wakatani/hasegawa-wakatani_example.ipynb)
    dx = ds["dx"].isel(x=0).values

    # Get rid of existing "x" coordinate, which is just the index values.
    ds = ds.drop("x")

    # Create a new coordinate, which is length in units of rho_s
    ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx[0])

    ### Unnormalise all spatial coordinates
    # Note Xarray won't let you do it inplace, hence this method
    ds = ds.assign_coords(x=ds["x"]*rho_s0)
    ds = ds.assign_coords(y=ds["y"]*rho_s0)
    ds = ds.assign_coords(z=ds["z"]*rho_s0)
    
    ## Unnormalise grid spacing
    ds["dx"] *= rho_s0
    ds["dy"] *= rho_s0
    ds["dz"] *= rho_s0

    ### Unnormalise time
    ds = ds.assign_coords(t=ds["t"]*(1/omega_ci))
    
    ### Add useful parameters to metadata
    ds.metadata["T0"] = T0
    ds.metadata["n0"] = n0
    ds.metadata["B"] = B
    ds.metadata["Z"] = Z
    ds.metadata["rho_s0"] = rho_s0
    
    
    print(f"Domain extents:")
    print(f"X: {ds['x'].min().values*1e3:.1f} - {ds['x'].max().values*1e3:.1f} mm")
    print(f"Y: {ds['y'].min().values*1e3:.1f} - {ds['y'].max().values*1e3:.1f} mm")
    print(f"Z: {ds['z'].min().values*1e3:.1f} - {ds['z'].max().values*1e3:.1f} mm")
    print(f"\nTime simulated: {ds['t'].max().values:.1e} s")
    
    return ds

def get_alpha_kappa(
    T0 = 50,            # Temperature [eV]
    n0 = 1e19,          # Density [m^-3]
    B = 0.5,            # Magnetic field [T]
    Z = 1,              # Ion charge
    lambda_n = 0.015,     # SOL radial density decay length
    dy = 0.2,
    ny = 64
    ):
    
    ### Constants
    Mp = 1.6726219e-27 # Proton mass [kg]
    Mi = Mp*2  # Ion mass of deuterium [kg]
    Me = 9.1093837e-31 # Electron mass [kg]
    qe = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    e0 = 8.854187817e-12 # Vacuum permittivity [F m^-1]
    kb = 1.38064852e-23 # Boltzmann constant [J K^-1]

    ### Normalisations
    omega_ci = qe * B / Mi   # ion gyrofrequency [s^-1]
    omega_ce = qe * B / Me   # electron gyrofrequency [s^-1]
    Cs0 = np.sqrt( (qe * T0) / Mi)   # Speed [m s^-1]
    rho_s0 = np.sqrt(Mi*T0/(qe*B))  # Distance, from hybrid Larmor radius [m]

    ## From Hermes-3 manual: 
    # https://hermes3.readthedocs.io/en/latest/components.html#collisions
    ## Originally from NRL formulary but converted to SI from CGS, hence different constants:
    # https://library.psfc.mit.edu/catalog/online_pubs/NRL_FORMULARY_19.pdf

    if T0 < 0.1 or n0 < 1e10:
        clog = 10   
    elif T0*Me/Mi < T0 and T0 < 10*Z**2:
        clog = 30 - 0.5*np.log(n0) + 1.5*np.log(T0)
    elif T0*Me/Mi < 10*Z**2 and 10*Z**2 <= T0:    # <--------- This one for T0=50, n0=1e19
        clog = 31 - 0.5*np.log(n0) + np.log(T0)
    elif T0 < T0*Me/Mi:
        clog = 23 - 0.5*np.log(n0) + 1.5*np.log(T0) - np.log(Z*Mi)
    else:
        raise Exception(f"T0 {T0} and n0 {n0} not in range of NRL formulary")
            

    # Collision time from Fitzpatrick https://farside.ph.utexas.edu/teaching/plasma/lectures1/node35.html
    T0_celsius = T0 * qe / kb
    T0_joules = T0 * qe

    eta = (np.pi * qe**2 * np.sqrt(Me)) / (4 * np.pi * e0)**2 * (T0 * qe)**(3/2) * clog
    eta_spitzer = 5.2e-5 * (Z*clog) / (T0**(3/2))  # T0 in eV

    Lpar = dy * ny * rho_s0
    k = 2*np.pi / Lpar    # Parallel wavenumber, should be y-direction representative wavenumber (2pi/lpar)
    # k = 1
    alpha = (T0 * 1**2) / (n0 * qe * eta_spitzer * omega_ci)
    alpha_2d = (T0 * k**2) / (n0 * qe * eta_spitzer * omega_ci)
    
    kappa = rho_s0 / lambda_n
    print(f"# T0: {T0}")
    print(f"# n0: {n0}")
    print(f"# B: {B}")
    print(f"# Z: {Z}")
    print(f"# lambda_n: {lambda_n:.6f}")
    print(f"# dy: {dy}")
    print(f"# ny: {ny}")
    print(f"# rho_s0: {rho_s0:.6f}")
    print(f"# Lpar: {Lpar:.4f}")
    print(f"# alpha: {alpha:.2e}")
    print(f"# alpha 2D: {alpha_2d:.2e}")
    print(f"# kappa: {kappa:.4f}")
    return alpha, kappa
    
def constants(name):
    
    out = dict()
    
    ### Constants
    out["Mp"] = 1.6726219e-27 # Proton mass [kg]
    out["Mi"] = out["Mp"]*2  # Ion mass of deuterium [kg]
    out["Me"] = 9.1093837e-31 # Electron mass [kg]
    out["qe"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    out["e0"] = 8.854187817e-12 # Vacuum permittivity [F m^-1]
    out["kb"] = 1.38064852e-23 # Boltzmann constant [J K^-1]
    
    return out[name]