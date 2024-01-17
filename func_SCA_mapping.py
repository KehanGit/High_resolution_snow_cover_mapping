import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.enums import MergeAlg
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib as mpl
import string
from multiprocessing import Pool

import fiona
from subprocess import Popen

import joblib
import time
import os
import glob
import fiona
from datetime import datetime
from itertools import cycle
import seaborn as sns
import sys
import json
import requests
from zipfile import ZipFile

from planet.api.auth import find_api_key
from planet.api.utils import strp_lenient

import earthpy.plot as ep
import earthpy.spatial as es
from natsort import natsorted
# from moviepy.editor import *

import rioxarray # for the extension to load
import xarray as xr
from tenacity import retry
from shapely.geometry import Polygon

from sklearn.model_selection import train_test_split
# evaluate random forest algorithm for classification
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from matplotlib import gridspec
from descartes import PolygonPatch
from matplotlib.ticker import FormatStrFormatter

import statsmodels.api as sm
from scipy import stats


import warnings
warnings.filterwarnings('ignore')

def feature_to_df2(row):
    # df = data.iloc[i]
    # sca = df["MeanNDSI"]
    sca = row
    sca = sca[1: (len(sca)-1)] # remove "{""}"
    sca_list = sca.split(",")

    date_list = [date[1:8] for date in sca_list]
    date_list[0] = sca_list[0][0:7]

    sca_res = [s[9:len(s)] for s in sca_list]
    sca_res[0] = sca_list[0][8:len(sca_list[0])]

    df_temp = pd.DataFrame(data = {"date" : date_list, "sca": sca_res})

    return df_temp




# read overall evaluation data for overall only, instrument info, remove low quanlity images
def read_validation(dir_root, tag):
    fn_4b = glob.glob(dir_root + tag)
    region_ID = [l.split('/')[-1].split('_')[0] for l in fn_4b]

    df = pd.DataFrame()
    for i in range(len(fn_4b)):
        data = pd.read_csv(fn_4b[i])
        data['region_ID'] = region_ID[i]
        df = pd.concat([df, data])

    # print(df.columns)
    df['datestr'] = [f.split('/')[-1].split('_')[0] for f in df['file_SCA']]
    df['group_ID'] = df['region_ID'] + df['datestr']

    df_overall = df[df.label == 'overall']
    df_overall['image_id'] = [i.split('/')[-1].split('_3B')[0] for i in df_overall['file_SCA']]
    images_quality = pd.read_csv('~/Data/project/pc2_snow_cover_mapping_meadows/pipeline/data/results/image_info/img_quality.csv')
    df_overall = df_overall.merge(images_quality, left_on = 'image_id', right_on = 'image')
    # combine instrument info 
    dir_meta_out = '/data0/kyang33/project/pc2_snow_cover_mapping_meadows/pipeline/data/results/validation/instruments.txt'
    df_meta = pd.read_csv(dir_meta_out)
    df_meta['image_id'] = [i.split('_m')[0] for i in df_meta['file']]
    df_overall = df_overall.merge(df_meta, left_on = 'image_id', right_on = 'image_id')

    # 1) only use images having standard quanlity; EUCHDB does not applied.
    df_standard = df_overall[(df_overall['quality_category'] == 'standard') | (df_overall['region_ID'] == 'EUCHDB')]

    # 2) remove low coverage date and the images not on the same date
    df_standard = df_standard[df_standard['datestr'] != '20190503'] # TE low coverage, very small overlaped area
    df_standard = df_standard[df_standard['datestr'] != '20190706'] # having images on 0705, so removed 0706 images (same day)


    # 3) remove images with low qualities by visually checking
    images_remove = pd.read_csv('/data0/kyang33/project/pc2_snow_cover_mapping_meadows/pipeline/data/results/validation/CA_remove_images_visual_examine.txt', header = None)
    images_remove = [a.split('_3B')[0] for a in images_remove.iloc[:,0]]

    df_standard_remove = df_standard[~df_standard['image_id'].isin(images_remove)]
    df_standard_remove = df_standard_remove.sort_values('f1')
    # 4) save results
    df_standard_remove.to_csv('~/Data/project/pc2_snow_cover_mapping_meadows/pipeline/check_data.csv')

    df_standard_remove['FSCA_predicted'] = (df_standard_remove['TP'] + df_standard_remove['FP'])/(df_standard_remove['TP'] + df_standard_remove['FP']+df_standard_remove['FN'] + df_standard_remove['TN'])
    df_standard_remove['FSCA_obs'] = (df_standard_remove['TP'] + df_standard_remove['FN'])/(df_standard_remove['TP'] + df_standard_remove['FP']+df_standard_remove['FN'] + df_standard_remove['TN'])
    df_standard_remove['PBIAS'] = (df_standard_remove['FSCA_predicted'] - df_standard_remove['FSCA_obs'])/df_standard_remove['FSCA_obs']*100
    df_standard_remove['month'] = [d[4:6] for d in df_standard_remove['datestr']]
    return df_standard_remove

def read_validation_label(dir_root, tag, read_label):
    fn_4b = glob.glob(dir_root + tag)

    region_ID = [l.split('/')[-1].split('_')[0] for l in fn_4b]
    # print(region_ID)

    df = pd.DataFrame()
    for i in range(len(fn_4b)):
        data = pd.read_csv(fn_4b[i])
        data['region_ID'] = region_ID[i]
        df = pd.concat([df, data])


    # print(df.columns)
    df['datestr'] = [f.split('/')[-1].split('_')[0] for f in df['file_SCA']]
    # df['datestr'] = np.where(df['datestr']=='20190706', '20190705', df['datestr'])
    df['group_ID'] = df['region_ID'] + df['datestr']
    # print(df['datestr'].unique())

    df_overall = df[df.label == read_label]
    # df_overall = df[df.label == '1']
    df_overall['image_id'] = [i.split('/')[-1].split('_3B')[0] for i in df_overall['file_SCA']]
    images_quality = pd.read_csv('~/Data/project/pc2_snow_cover_mapping_meadows/pipeline/data/results/image_info/img_quality.csv')
    df_overall = df_overall.merge(images_quality, left_on = 'image_id', right_on = 'image')
    # combine instrument info 
    dir_meta_out = '/data0/kyang33/project/pc2_snow_cover_mapping_meadows/pipeline/data/results/validation/instruments.txt'
    df_meta = pd.read_csv(dir_meta_out)
    df_meta['image_id'] = [i.split('/')[-1].split('_meta')[0] for i in df_meta['file']]
    df_overall = df_overall.merge(df_meta, left_on = 'image_id', right_on = 'image_id')

    # 1) only use images having standard quanlity; EUCHDB does not applied.
    df_standard = df_overall[(df_overall['quality_category'] == 'standard') | (df_overall['region_ID'] == 'EUCHDB')]

    # 2) remove low coverage date and the images not on the same date
    df_standard = df_standard[df_standard['datestr'] != '20190503'] # TE low coverage, very small overlaped area
    df_standard = df_standard[df_standard['datestr'] != '20190706'] # having images on 0705, so removed 0706 images (same day)


    # 3) remove images with low qualities by visually checking
    images_remove = pd.read_csv('/data0/kyang33/project/pc2_snow_cover_mapping_meadows/pipeline/data/results/validation/CA_remove_images_visual_examine.txt', header = None)
    images_remove = [a.split('_3B')[0] for a in images_remove.iloc[:,0]]

    df_standard_remove = df_standard[~df_standard['image_id'].isin(images_remove)]
    df_standard_remove = df_standard_remove.sort_values('f1')
    # 4) save results
    df_standard_remove.to_csv('~/Data/project/pc2_snow_cover_mapping_meadows/pipeline/check_data.csv')

    df_standard_remove['FSCA_predicted'] = (df_standard_remove['TP'] + df_standard_remove['FP'])/(df_standard_remove['TP'] + df_standard_remove['FP']+df_standard_remove['FN'] + df_standard_remove['TN'])
    df_standard_remove['FSCA_obs'] = (df_standard_remove['TP'] + df_standard_remove['FN'])/(df_standard_remove['TP'] + df_standard_remove['FP']+df_standard_remove['FN'] + df_standard_remove['TN'])
    df_standard_remove['PBIAS'] = (df_standard_remove['FSCA_predicted'] - df_standard_remove['FSCA_obs'])/df_standard_remove['FSCA_obs']*100
    df_standard_remove['month'] = [d[4:6] for d in df_standard_remove['datestr']]
    # print(df_standard_remove.groupby('group_ID_x')['f1'].describe())
    # df_standard_remove.groupby('group_ID_x')['f1'].describe()
    return df_standard_remove
    # df_standard_remove.describe()
def cal_consecutive_num(xx):
    # calculate the consecutvive number of each element
    # xx = pd.DataFrame()
    # xx['num'] = [0,0,0,1,1,1,0,0,1,0,1,1,1,1]
    cnt = []
    num = []
    N = 1
    for i in range(xx['num'].shape[0]):
        if i == xx['num'].shape[0]-1:
            num.append(xx['num'][i])
            cnt.append(N)
        elif xx['num'][i+1] == xx['num'][i]: 
            N = N+1 
        else:
            num.append(xx['num'][i])
            cnt.append(N)
            N = 1

    res = pd.DataFrame()
    res['num'] = num
    res['cnt'] = cnt
    return res

# cal_consecutive_num(xx)


def split_raster_multipoly(dir_shp, dir_separated, file_input, dir_output, tag_output):
    # root directory
    # dir_root = '/Users/kehanyang/Documents/resarch/pc2_meadows/data/ASO/canopy/'
    # input shapefile with multiple polygons used to split the raster
    # dir_shp = dir_root + 'geojson/TE_clip_poly.shp'
    # directory of splited shapefiles
    # dir_separated = dir_root + 'geojson/separated/TE/'
    # input CHM file
    # file_input = dir_root + 'ASO_3M_CHM_USCATB_20140827.tif'
    # directory of output splited files
    # dir_output = dir_root + 'split/TE/'
    # the lable of output file
    # tag_output = 'TE_CHM_split_'
    if not os.path.exists(dir_separated): os.mkdir(dir_separated)
    if not os.path.exists(dir_output): os.mkdir(dir_output)

    with fiona.open(dir_shp, 'r') as dst_in:
        for index, feature in enumerate(dst_in):
            with fiona.open(dir_separated + 'polygon{}.shp'.format(index), 'w', **dst_in.meta) as dst_out:
                dst_out.write(feature)

    polygons = glob.glob(dir_separated + '*.shp')  ## Retrieve all the .shp files

    N = 1
    for polygon in polygons:
        command = 'gdalwarp -dstnodata -9999 -cutline {} ' \
                '-crop_to_cutline -of GTiff {} {}'.format(polygon, file_input, dir_output + tag_output + str(N) + '.tiff')
        Popen(command, shell=True)
        N = N + 1

        

def validation_meadow(dir_aso, fn_pred, th, sub_shp_meadows):
    num = []
    label = []
    files = []
    OBJECTID = []
    validation_res = pd.DataFrame()
    file_num = []
    ds_aso = xr.open_rasterio(dir_aso)
    for i in range(len(fn_pred)):
        print(fn_pred[i])
        ds_planet = xr.open_rasterio(fn_pred[i])
        #clip aso data to planet extent and projection
        ds_aso_clip = ds_aso.rio.reproject_match(ds_planet)

        # subset meadows shapefile to the planet extent
        xmin, ymin, xmax, ymax = ds_planet.rio.bounds()
        bounds = Polygon( [(xmin,ymin), (xmin, ymax), (xmax, ymax), (xmax,ymin)] )
        sub_shp_meadows['within1'] = sub_shp_meadows['geometry'].within(bounds)
        sub_shp_meadows2 = sub_shp_meadows[sub_shp_meadows['within1']]
        if sub_shp_meadows2.shape[0] > 0:
            # loop each meadow polygon and do evaluation
            for j in range(sub_shp_meadows2.shape[0]):
            # for j in range(2):
                one_meadow = sub_shp_meadows2.iloc[j:j+1]
                #subset planet and aso to meadow extent
                ds_planet_sub = ds_planet.rio.clip(one_meadow['geometry'], ds_planet.rio.crs)
                ds_aso_sub = ds_aso_clip.rio.clip(one_meadow['geometry'], ds_aso_clip.rio.crs)

                df = pd.DataFrame()
                df['ASO_SD'] = np.reshape(ds_aso_sub.values, (np.product(ds_aso_sub.values.shape),))
                df['predict'] = np.reshape(ds_planet_sub.values, (np.product(ds_planet_sub.values.shape),))
                df['obs'] = np.where(df['ASO_SD'] > th, 1.0, 0.0)
                df = df[df['ASO_SD'] != -9999]
                df = df.dropna()
                if df.shape[0]>0:
                    files.append(fn_pred[i])
                    OBJECTID.append(one_meadow['OBJECTID'])
                    # print(one_meadow['OBJECTID'])
                    num.append(df.shape[0])
                    validation_res=pd.concat([validation_res, calculate_metrics(df)])
                    file_num.append(i)

    validation_res['th'] = th
    validation_res['pixel_num'] = num
    validation_res['file_SCA'] = files
    validation_res['file_ASO'] = dir_aso
    validation_res['file_num'] = file_num
    validation_res['OBJECTID'] = [i.to_list()[0] for i in OBJECTID]
    
    return(validation_res)

def plot_validation_DCE(validation_res, h_loc):
    validation_res1 = validation_res[validation_res.label != 'overall']
    # validation_res1 = validation_res
    x = validation_res1.label.to_list()
    y = validation_res1.f1.to_list()
    x = [str(i) for i in x]
    c = validation_res1['file_num'].to_list()
    # colors = [np.where(i == 0, 'img1', 'img2').item() for i in c]

    plt.figure(figsize=(12, 6))
    sns.pointplot(x = x,y=y, hue=c,alpha=0.9)
    sns.lineplot(x = x,y=y, hue=c,alpha=0.9)
    plt.axvline(5.5, linestyle = 'dashed', color = 'red', alpha = 0.5)
    plt.text(9.7,1.0, 'DCE=0', color = 'red')
    plt.xticks(rotation = 90)
    plt.show()

def validation_dce(ds_aso, ds_dce, fn_pred, th, gap, gmin, gmax):
    num = []
    label = []
    files = []
    validation_res = pd.DataFrame()
    file_num = []
    
    for i in range(len(fn_pred)):
        ds_planet = xr.open_rasterio(fn_pred[i])
        ds_aso_clip = ds_aso.rio.reproject_match(ds_planet)
        ds_dce_clip = ds_dce.rio.reproject_match(ds_planet)

        df = pd.DataFrame()
        df['ASO_SD'] = np.reshape(ds_aso_clip.values, (np.product(ds_aso_clip.values.shape),))
        df['predict'] = np.reshape(ds_planet.values, (np.product(ds_planet.values.shape),))
        df['obs'] = np.where(df['ASO_SD'] > th, 1.0, 0.0)
        df['dce'] = np.reshape(ds_dce_clip.values, (np.product(ds_dce_clip.values.shape),))
        
        df = df[df['ASO_SD'] != -9999]
        df = df[df['dce'] > -9999]
        df = df.dropna()

        tag = 0
        label_ranges = []
        for m in range(gmin, gmax, gap):
            lmin = m
            lmax = m+gap
            if tag == 0:
                df['label'] = np.where((df['dce'] > lmin) & (df['dce'] <= lmax), str(lmin)+'_'+str(lmax), np.nan)
                df['label'] = np.where(df['dce'] <= lmin, '<='+str(lmin), df['label'])
                # print('1 -- '+ str(m) + ':' + str(lmin)+'_'+str(lmax))
                tag = tag + 1
                label_ranges.append(str(lmin)+'_'+str(lmax))
            elif lmin < gmax:
                df['label'] = np.where((df['dce'] > lmin) & (df['dce'] <= lmax), str(lmin)+'_'+str(lmax), df['label'])
                # print('2 -- '+ str(m) + ':' + str(lmin)+'_'+str(lmax))
                label_ranges.append(str(lmin)+'_'+str(lmax))
            if lmax +gap > gmax:
                df['label'] = np.where(df['dce'] > lmax, '>'+str(lmax), df['label'])
                # print('3 -- '+ str(m) + ':' + '>'+str(lmax))
                label_ranges.append( '>'+str(lmax))


        label.append('overall')
        num.append(df.shape[0])
        files.append(fn_pred[i])
        validation_res=pd.concat([validation_res, calculate_metrics(df)])
        file_num.append(i)
        for j in label_ranges:
            df_cal = df[df['label'] == j]
            num.append(df_cal.shape[0])
            label.append(j)
            files.append(fn_pred[i])
            validation_res=pd.concat([validation_res, calculate_metrics(df_cal)])
            file_num.append(i)

    validation_res['th'] = th
    validation_res['label']=label
    validation_res['pixel_num'] = num
    validation_res['file_SCA'] = files
    # validation_res['file_ASO'] = dir_aso
    validation_res['file_num'] = file_num
    
    return(validation_res)




def validation_sca(ds_aso, fn_pred, dir_out, th, dir_aso):
    label = []
    files = []
    num = []
    validation_res = pd.DataFrame()
   
    for i in range(len(fn_pred)):
        print(fn_pred[i])
        ds_planet = xr.open_rasterio(fn_pred[i])
        ds_aso_clip = ds_aso.rio.reproject_match(ds_planet)

        df = pd.DataFrame()
        df['ASO_SD'] = np.reshape(ds_aso_clip.values, (np.product(ds_aso_clip.values.shape),))
        df['predict'] = np.reshape(ds_planet.values, (np.product(ds_planet.values.shape),))
        df['obs'] = np.where(df['ASO_SD'] > th, 1.0, 0.0)

       
        df = df[df['ASO_SD'] != -9999]
        df = df[np.isnan(df.obs) == False]
        df = df.dropna()
        

        label.append('overall')
        num.append(df.shape[0])
        files.append(fn_pred[i])
        validation_res=pd.concat([validation_res, calculate_metrics(df)])

    validation_res['th'] = th
    validation_res['label']=label
    validation_res['pixel_num'] = num
    validation_res['file_SCA'] = files
    validation_res['file_ASO'] = dir_aso
    print('Save file to: ' + dir_out)
    validation_res.to_csv(dir_out)
    
    
    
def read_hls(dir_download_hls):
    stack_df = pd.read_csv(dir_download_hls)
    stack_df = stack_df.loc[~stack_df['date'].isna(), :]
    stack_df.reset_index(inplace=True)
    # subset the s3 links by band
    header = ['label', 'L30_band', 'S30_band', 'read']
    data = [
        ['coastal_aerosol', 'B01', 'B01', False],
        ['blue', 'B02', 'B02', True],
        ['green', 'B03', 'B03', True],
        ['red', 'B04', 'B04', True],
        ['red-edge_1', None, 'B05', False],
        ['red-edge_2', None, 'B06', False],
        ['red-edge_3', None, 'B07', False],
        ['nir_broad', None, 'B08', False],
        ['nir', 'B05', 'B8A', True],
        ['swir_1', 'B06', 'B11', True],
        ['swir_2', 'B07', 'B12', True],
        ['water_vapor', None, 'B09', False],
        ['cirrus', 'B09', 'B10', False],
        ['thermal_infrared_1', 'B10', None, False],
        ['thermal_infrared_2', 'B11', None, False],
        ['fmask', 'Fmask', 'Fmask', True]
    ]

    band_df = pd.DataFrame(data, columns=header)
    # print(band_df)

    print('Total images: ' + str(len([f for f in stack_df.S3_links if f.endswith('B01.tif')])))

    chunks=dict(band=1, x=256, y=256)


    hls_ds = None
    print(band_df.shape[0])

    for i in range(0, band_df.shape[0]):
        if band_df.loc[i, 'read'] == True:
            # subset stack for links for each band
            band_stack = stack_df.loc[
            ((stack_df['band'] == band_df.loc[i,'L30_band']) & (stack_df['sensor'] == 'L30')) |
            ((stack_df['band'] == band_df.loc[i,'S30_band']) & (stack_df['sensor'] == 'S30')), :]

            # create the time index
            band_time = [datetime.strptime(str(t), '%Y%jT%H%M%S') for t in band_stack['date']]
            xr.Variable('time', band_time)

            s3_links = band_stack['S3_links']
            local_linls = band_stack['local_links']

            # get the band label
            band_label = band_df.loc[i, 'label']

            # open the links?
            hls_ts_da = xr.concat([rioxarray.open_rasterio(f, chunks=chunks, parse_coordinates = True).squeeze('band', drop=True) for f in local_linls], dim=band_time)
            hls_ts_da.rename({'concat_dim':'time'})

            if hls_ds is None:
                hls_ds = xr.Dataset({band_label: hls_ts_da})
            else:
                hls_ds[band_label] = hls_ts_da
                
    return hls_ds




                

def zipfilesinfolder(dirName, dir_zip):
    # dirName = '/data0/kyang33/project/pc2_snow_cover_mapping_meadows/pipeline/HLS_data/RF_train/'
    # create a ZipFile object
    with ZipFile(dir_zip, 'w') as zipObj:
       # Iterate over all the files in directory
       for folderName, subfolders, filenames in os.walk(dirName):
               for filename in filenames:
                   #create complete filepath of file in directory
                   filePath = os.path.join(folderName, filename)
                   # Add file to zip
                   zipObj.write(filePath, os.path.basename(filePath))

# rasterize the ROI for model training
def vector_rasterize(dir_vector, dir_raster, dir_out, flag_output):
    vector = gpd.read_file(dir_vector)
    # Get list of geometries for all features in vector file
    geom = [shapes for shapes in vector.geometry]

    # Open example raster
    raster = rasterio.open(dir_raster)
    
    # reproject vector to raster
    vector = vector.to_crs(raster.crs)

    # create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value = ((geom,value) for geom, value in zip(vector.geometry, vector['label']))

    # Rasterize vector using the shape and transform of the raster
    rasterized = features.rasterize(geom_value,
                                    out_shape = raster.shape,
                                    transform = raster.transform,
                                    all_touched = True,
                                    fill = 9,   # background value
                                    merge_alg = MergeAlg.replace,
                                    dtype = np.float32)


    if flag_output:
        with rasterio.open(
                dir_out, "w",
                driver = "GTiff",
                transform = raster.transform,
                dtype = rasterio.float32,
                count = 1,
                width = raster.width,
                height = raster.height) as dst:
            dst.write(rasterized, indexes = 1)
    return rasterized

# run model prediction
def run_sca_prediction(f_raster, file_out, nodata_flag, model):
    ndvi_out = os.path.dirname(file_out) + '/' +os.path.basename(file_out)[0:-8] + '_NDVI.tif'
    if not os.path.exists(ndvi_out):
        print(ndvi_out)
        print('Start to predict:'.format(), os.path.basename(f_raster))

        with rasterio.open(f_raster, 'r') as ds:
            arr = ds.read()  # read all raster values

        print("Image dimension:".format(), arr.shape)  # 
        X_img = pd.DataFrame(arr.reshape([4,-1]).T)
        X_img = X_img/10000 # scale surface reflectance to 0-1
        X_img.columns = ['blue','green','red','nir']
        X_img['ndvi'] = (X_img['nir']-X_img['red'])/(X_img['nir']+X_img['red'])

        X_img[X_img['ndvi']< -1.0]['ndvi'] = -1.0
        X_img[X_img['ndvi']> 1.0]['ndvi'] = 1.0
        X_img['ndvi'] = np.where(np.isfinite(X_img['ndvi']), X_img['ndvi'], 0) # fill nan by NA
        

        # model.fit(X,y)
        y_img = model.predict(X_img)
        X_img['nodata_flag'] = np.where(X_img['blue']==0, -1, 1)
        X_img['ndvi_nan'] = (X_img['nir']-X_img['red'])/(X_img['nir']+X_img['red'])
        

        out_img = pd.DataFrame()
        out_img['label'] = y_img
        out_img['ndvi'] = (X_img.nir - X_img.red)/(X_img.nir + X_img.red)
        out_img['label'] = np.where(X_img['nodata_flag'] == -1, np.nan, out_img['label'])
        out_img['label'] = np.where(np.isnan(X_img['ndvi_nan']), np.nan, out_img['label'])
        
        
        # Reshape our classification map
        img_prediction = out_img['label'].to_numpy().reshape(arr[0,:, :].shape)
        img_ndvi = out_img['ndvi'].to_numpy().reshape(arr[0,:, :].shape)


        # save image to file_out

        with rasterio.open(
                file_out, "w",
                driver = "GTiff",
                transform = ds.transform,
                dtype = rasterio.float32,
                count = 1,
                crs = ds.crs,
                width = ds.width,
                height = ds.height) as dst:
            dst.write(img_prediction, indexes = 1)

        with rasterio.open(
                ndvi_out, "w",
                driver = "GTiff",
                transform = ds.transform,
                dtype = rasterio.float32,
                count = 1,
                crs = ds.crs,
                width = ds.width,
                height = ds.height) as dst:
            dst.write(img_ndvi, indexes = 1)

            
            # run model prediction
def run_sca_prediction_band(f_raster, file_out, nodata_flag, model):
    ndvi_out = os.path.dirname(file_out) + '/' +os.path.basename(file_out)[0:-8] + '_NDVI.tif'
    if not os.path.exists(ndvi_out):
        print(ndvi_out)
        print('Start to predict:'.format(), os.path.basename(f_raster))

        with rasterio.open(f_raster, 'r') as ds:
            arr = ds.read()  # read all raster values

        print("Image dimension:".format(), arr.shape)  # 
        X_img = pd.DataFrame(arr.reshape([4,-1]).T)
        X_img = X_img/10000 # scale surface reflectance to 0-1
        X_img.columns = ['blue','green','red','nir']
        # model.fit(X,y)
        y_img = model.predict(X_img)
        
        X_img['ndvi'] = (X_img['nir']-X_img['red'])/(X_img['nir']+X_img['red'])
        X_img[X_img['ndvi']< -1.0]['ndvi'] = -1.0
        X_img[X_img['ndvi']> 1.0]['ndvi'] = 1.0
        X_img['ndvi'] = np.where(np.isfinite(X_img['ndvi']), X_img['ndvi'], 0) # fill nan by NA
        

        
        X_img['nodata_flag'] = np.where(X_img['blue']==0, -1, 1)
        X_img['ndvi_nan'] = (X_img['nir']-X_img['red'])/(X_img['nir']+X_img['red'])
        

        out_img = pd.DataFrame()
        out_img['label'] = y_img
        out_img['ndvi'] = (X_img.nir - X_img.red)/(X_img.nir + X_img.red)
        out_img['label'] = np.where(X_img['nodata_flag'] == -1, np.nan, out_img['label'])
        out_img['label'] = np.where(np.isnan(X_img['ndvi_nan']), np.nan, out_img['label'])
        
        
        # Reshape our classification map
        img_prediction = out_img['label'].to_numpy().reshape(arr[0,:, :].shape)
        img_ndvi = out_img['ndvi'].to_numpy().reshape(arr[0,:, :].shape)


        # save image to file_out

        with rasterio.open(
                file_out, "w",
                driver = "GTiff",
                transform = ds.transform,
                dtype = rasterio.float32,
                count = 1,
                crs = ds.crs,
                width = ds.width,
                height = ds.height) as dst:
            dst.write(img_prediction, indexes = 1)

        with rasterio.open(
                ndvi_out, "w",
                driver = "GTiff",
                transform = ds.transform,
                dtype = rasterio.float32,
                count = 1,
                crs = ds.crs,
                width = ds.width,
                height = ds.height) as dst:
            dst.write(img_ndvi, indexes = 1)

# run model prediction
def run_sca_prediction_fusion(dir_raster, dir_out, nodata_flag, model):
    for f in glob.glob(dir_raster + '/*.tif', recursive = True):
        file_out = dir_out + '/'+ os.path.basename(f)[0:-4] + '_SCA.tif'
        ndvi_out = dir_out + '/' + os.path.basename(f)[0:-4] + '_NDVI.tif'
        # print(file_out)
        # if file exist, next
        if os.path.exists(file_out):
            print (os.path.basename(f) + " SCA exist!")
        else:
            print('Start to predict:'.format(), os.path.basename(f))

        with rasterio.open(f, 'r') as ds:
            arr = ds.read()  # read all raster values

        print("Image dimension:".format(), arr.shape)  # 
        X_img = pd.DataFrame(arr.reshape([4,-1]).T)
        X_img = X_img/10000 # scale surface reflectance to 0-1
        X_img.columns = ['blue','green','red','nir']

        # model.fit(X,y)
        y_img = model.predict(X_img)
        X_img['nodata_flag'] = np.where(X_img['blue']==0, -1, 1)

        out_img = pd.DataFrame()
        out_img['label'] = y_img
        out_img['ndvi'] = (X_img.nir - X_img.red)/(X_img.nir + X_img.red)
        out_img['nodata_flag'] = X_img['nodata_flag']
        out_img['label'] = np.where(out_img['nodata_flag'] == -1, np.nan, out_img['label'])
        # Reshape our classification map
        img_prediction = out_img['label'].to_numpy().reshape(arr[0,:, :].shape)
        img_ndvi = out_img['ndvi'].to_numpy().reshape(arr[0,:, :].shape)


        # save image to file_out

        with rasterio.open(
                file_out, "w",
                driver = "GTiff",
                transform = ds.transform,
                dtype = rasterio.float32,
                count = 1,
                crs = ds.crs,
                width = ds.width,
                height = ds.height) as dst:
            dst.write(img_prediction, indexes = 1)

        with rasterio.open(
                ndvi_out, "w",
                driver = "GTiff",
                transform = ds.transform,
                dtype = rasterio.float32,
                count = 1,
                crs = ds.crs,
                width = ds.width,
                height = ds.height) as dst:
            dst.write(img_ndvi, indexes = 1)



# run model prediction
def run_sca_prediction_meadows(dir_raster, dir_out, nodata_flag, model):
    subfolders = [ f.path for f in os.scandir(dir_raster) if f.is_dir() ]
    ids = [x.split('/')[-1] for x in subfolders]
    for i in range(len(ids)):
        if not os.path.exists(dir_out+ids[i]): os.makedirs(dir_out+ids[i])
    
        for f in glob.glob(dir_raster + ids[i] + '/./**/*SR*.tif', recursive = True):
            file_out = dir_out + ids[i] + '/' + os.path.basename(f)[0:-4] + '_SCA.tif'
            ndvi_out = dir_out + ids[i] + '/' + os.path.basename(f)[0:-4] + '_NDVI.tif'
            print(file_out)
            # if file exist, next
            if os.path.exists(file_out):
                print (os.path.basename(f) + " SCA exist!")
            else:
                print('Start to predict:'.format(), os.path.basename(f))

                with rasterio.open(f, 'r') as ds:
                    arr = ds.read()  # read all raster values

                print("Image dimension:".format(), arr.shape)  # 
                X_img = pd.DataFrame(arr.reshape([4,-1]).T)
                X_img = X_img/10000 # scale surface reflectance to 0-1
                X_img.columns = ['blue','green','red','nir']

                # model.fit(X,y)
                y_img = model.predict(X_img)
                X_img['nodata_flag'] = np.where(X_img['blue']==0, -1, 1)

                out_img = pd.DataFrame()
                out_img['label'] = y_img
                out_img['ndvi'] = (X_img.nir - X_img.red)/(X_img.nir + X_img.red)
                out_img['nodata_flag'] = X_img['nodata_flag']
                out_img['label'] = np.where(out_img['nodata_flag'] == -1, np.nan, out_img['label'])
                # Reshape our classification map
                img_prediction = out_img['label'].to_numpy().reshape(arr[0,:, :].shape)
                img_ndvi = out_img['ndvi'].to_numpy().reshape(arr[0,:, :].shape)
                

                # save image to file_out

                with rasterio.open(
                        file_out, "w",
                        driver = "GTiff",
                        transform = ds.transform,
                        dtype = rasterio.float32,
                        count = 1,
                        crs = ds.crs,
                        width = ds.width,
                        height = ds.height) as dst:
                    dst.write(img_prediction, indexes = 1)
                
                with rasterio.open(
                        ndvi_out, "w",
                        driver = "GTiff",
                        transform = ds.transform,
                        dtype = rasterio.float32,
                        count = 1,
                        crs = ds.crs,
                        width = ds.width,
                        height = ds.height) as dst:
                    dst.write(img_ndvi, indexes = 1)

   
# calculate five evaluation metrics
def calculate_metrics(df):
    
    out = pd.DataFrame(data = {'precision': [0], 
                               'recall':[0], 
                               'f1':[0],
                               'sensitivity':[0],
                               'specificity':[0],
                               'balanced_accuracy':[0], 
                               'accuracy':[0],
                               'kappa':[0],
                               'TP':[0],
                               'TN':[0],
                               'FP':[0],
                               'FN':[0]})
    
    # true positive and true negative
    subdf = df[df.predict == df.obs] # true prediction
    TP = len(subdf[subdf.predict == 1].index) # true positive
    TN = len(subdf[subdf.predict == 0].index) # true negative
    

    # false positive and false negative
    subdf = df[df.predict != df.obs] # false prediction 
    FN = len(subdf[subdf.predict == 0].index)# false negative
    FP = len(subdf[subdf.predict == 1].index) # false positive

    
    if (TP + FP) == 0 or (TP + FN) == 0 or (TN + FP) == 0 or (TP+FN) == 0:
        return out
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)

        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        balanced_accuracy = (sensitivity+specificity)/2
        accuracy = (TP+TN)/(TP+TN+FP+FN)

        f1 =  0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        kappa = cohen_kappa_score(df.predict, df.obs)
        out = pd.DataFrame(data = {'precision': [precision], 
                                   'recall':[recall], 
                                   'f1':[f1],
                                   'sensitivity':[sensitivity],
                                   'specificity':[specificity],
                                   'balanced_accuracy':[balanced_accuracy], 
                                   'accuracy':[accuracy],
                                   'kappa':[kappa],
                                   'TP':[TP],
                                   'TN':[TN],
                                   'FP':[FP],
                                   'FN':[FN]})
        return out

#  clip raster to the regions within shapefile
def clip_raster_shp_1D(dir_inshp, dir_in_img, dir_out_img):
    """
    dir_inshp: the extent shapefile
    dir_in_img: the file needs to be croped 
    dir_out_img: output file directory
    """
    with fiona.open(dir_inshp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(dir_in_img,'r') as src:
        out_img, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta

    # Save clipped imagery
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir_out_img, "w", **out_meta) as dest:
        dest.write(out_img)
    return out_img

# calcualte mean SCA and NDVI for the meadow within the original meadow polygon
def F_calculate_regional_SCA_NDVI(Run_ID, dir_model_predict, file_meadow_extent, file_meadow_info, dir_result):
    
    """
    #' Run_ID: folder name
    #' dir_model_predict: SCA folder
    #' file_meadow_extent: original shapefile of meadow extent from US Davis
    #' file_meadow_info: csv file which has region_id and OBJ_id that can link them together
    #' dir_result: the output folder
    """
    
    dir_run = dir_result + Run_ID + '/'
    if not os.path.exists(dir_run): os.mkdir(dir_run)
    # read meadow extent 
    shp_meadow_extent = gpd.read_file(file_meadow_extent)
    meadow_info = pd.read_csv(file_meadow_info)

    dir_list = [f.path for f in os.scandir(dir_model_predict) if f.is_dir()]
    dir_list_folder = [f.name for f in os.scandir(dir_model_predict) if f.is_dir()]
     
    # for each OBJCTID (represent each meadow), there is a region_ID which is stored in the meadow_info file 
    for i in range(len(dir_list_folder)):
        region_id = dir_list_folder[i][0:(len(dir_list_folder[i])-5)]
        obj_ids = meadow_info['OBJECTID'][meadow_info['Region_ID'] == region_id]
       
        fn_sca = [f for f in glob.glob(dir_list[i]+ "/*SCA.tif")]
        fnname_sca = [os.path.basename(f) for f in glob.glob(dir_list[i]+ "/*SCA.tif")]
        
        if len(obj_ids) > 0:
            for k in range(len(obj_ids)):
                df_all = pd.DataFrame()
                outcsv = dir_run + dir_list_folder[i] + '_OBJID_' + str(obj_ids.values[k]) + '.csv'
                if os.path.exists(outcsv):
                    print(outcsv + ' exist!')
                else:
                    for j in range(len(fn_sca)):
                        r_sca = rasterio.open(fn_sca[j],'r') # 0 == snow-free, 1 == snow, 2 == mixed pixel
                        # get polygon for obj_id from shapefile 
                        obj_shp = shp_meadow_extent[shp_meadow_extent['OBJECTID'] == obj_ids.values[k]]
                        # need to reproject polygon to the crs of raster 
                        obj_shp_repo = obj_shp.to_crs(r_sca.crs)
                        r_sca_clip = mask(r_sca, obj_shp_repo.geometry, crop = True, filled = False)

                        # calculate the area of sca
                        snow_area = np.count_nonzero(r_sca_clip[0] == 1.0)* 3 * 3/1000000
                        non_snow_area = np.count_nonzero(r_sca_clip[0] == 0.0)* 3 * 3/1000000

                        #find ndvi for this image 
                        tag = fnname_sca[j][0: len(fnname_sca[j])-19] + '*NDVI.tif'
                        fn_ndvi = glob.glob(dir_list[i]+ '/'+tag)

                        if len(fn_ndvi)==1:
                            r_ndvi = rasterio.open(fn_ndvi[0],'r')
                            r_ndvi_clip = mask(r_ndvi, obj_shp_repo.geometry, crop = True)
                            # get the mean ndvi for the region
                            ndvi_mean = r_ndvi_clip[0].mean() 
                        else:
                            ndvi_mean = np.nan


                        temp = pd.concat([pd.DataFrame([fnname_sca[j][0:8]], columns = ['date']), 
                                          pd.DataFrame([snow_area], columns = ['snow_area_km2']),
                                          pd.DataFrame([non_snow_area], columns = ['no_snow_area_km2']),
                                          pd.DataFrame([ndvi_mean], columns = ['ndvi_mean']),
                                          pd.DataFrame([obj_ids.values[k]], columns = ['obj_id']),
                                          pd.DataFrame([region_id], columns = ['region_id']),
                                          pd.DataFrame([fnname_sca[j][0:(len(fnname_sca[j])-19)]], columns = ['image'])], axis = 1)

                        df_all = pd.concat([df_all, temp])
                        df_all.to_csv(outcsv, index = False)