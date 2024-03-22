# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## import libraries
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import scipy.io
import cartopy.crs as ccrs
from scipy import optimize
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import gridspec
import matplotlib.patches as patches





import matplotlib.colors as colors
import matplotlib

%config Inline.Backend.figure_format='retina'







## LOAD FILES

## load etopo
etopo=scipy.io.loadmat('etopo_15.mat')
etopo_lat = etopo['LAT']
etopo_lon = etopo['LON']
etopo_bed = etopo['topo_bed']

## load ice history
sl_hist = scipy.io.loadmat("ICE-PC2_gi36_ice6gbase.mat")
pc2_i = sl_hist['newice']

## load time
time = np.loadtxt("time_icepc2_gi36.txt")

## make x and y axis for sl
sl_lon = np.linspace(0,360,1024)
sl_lat = np.linspace(90,-90,512)
sl_grid = np.meshgrid(sl_lon,sl_lat)







## CALC LAT & LON DISTANCES

## calculate distance between lat in km
from math import radians, sin, cos, sqrt, atan2

def calculate_distance_lat(lat1, lat2):
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude values from degrees to radians
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    
    # Difference in latitudes
    delta_lat = lat2_rad - lat1_rad
    
    # Haversine formula to calculate distance
    a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(0)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

## calculate distance between lon in km
def calculate_distance_lon(lon1, lon2, latitude):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert longitude difference to radians
    delta_lon = radians(lon2 - lon1)

    # Calculate the distance using the formula for longitude distance at a specific latitude
    distance = R * cos(radians(latitude)) * delta_lon

    return distance

        
        
## creating lat dist array
lat_list = np.zeros((len(sl_lat)-1))

for each in np.arange(0, len(sl_lat)-1):
    lat_dist = calculate_distance_lat(sl_lat[each], sl_lat[each+1])
    lat_list[each] = lat_dist
    
    
## creating lon matrix
lon_mat = np.zeros((511, 1023))

for i in np.arange(0, len(sl_lat)-1):
    for j in np.arange(0, len(sl_lon)-1):
        lon_mat[i][:] = calculate_distance_lon(sl_lon[j], sl_lon[j+1], sl_lat[i])

for each in np.arange(0, len(lat_list)):
    lon_mat[each][:] = lat_list[each] * lon_mat[each][:]
    
    
## graph of area distribution 
plt.subplots(figsize = (12,8))
plt.pcolormesh(sl_lon, sl_lat, lon_mat, shading='auto')
plt.ylabel('lat')
plt.xlabel('lon')
plt.colorbar()
#plt.savefig('grid_area.png', dpi=400)


## store 
lat_vals = []
for each in np.arange(0, 511):
    lat_vals.append(lon_mat[each][0])
    
lat_vals = np.array(lat_vals)

lat_vals1 = np.append(lat_vals, 0)







