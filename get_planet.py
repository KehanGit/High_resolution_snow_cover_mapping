### Import libraries 
## env snow
### Import libraries
import os
import requests
import json
import sys
import time
import pathlib
import pandas as pd
import numpy as np
import pyproj
from shapely.geometry import shape
from shapely.ops import transform
from datetime import datetime
from datetime import timezone
from time import mktime
from planet.api.auth import find_api_key
from planet.api.utils import strp_lenient
from requests.auth import HTTPBasicAuth
# from tenacity import retry
from retrying import retry

## Get your API Key
try:
    PLANET_API_KEY = find_api_key() #remove find_api_key and place your api key like 'api-key'
except Exception as e:
    print("Failed to get Planet Key: Try planet init or install Planet Command line tool")
    sys.exit()

headers = {'Content-Type': 'application/json'}

# check if API key is valid 
response = requests.get('https://api.planet.com/compute/ops/orders/v2',auth=(PLANET_API_KEY, ""))
if response.status_code==200:
    print('Setup OK: API key valid')
else:
    print(f'Failed with response code {response.status_code}: reinitialize using planet init')

#### Import your geometries
# In this case a geojson file is obtained for an AOI form geojson.io.
def read_geom(geom):
    with open(geom) as f:
        bbox = json.load(f)['features'][0]['geometry']
        return bbox

#### Setup Filters for parsing through data
# This step allows you to pass the geometry filter along with date range and cloud cover filter.

def time2utc(st):
    st_time = strp_lenient(st)
    if st_time is not None:
        dt_ts = datetime.fromtimestamp(time2epoch(st_time), tz=timezone.utc)
        return dt_ts.isoformat().replace("+00:00", "Z")
    else:
        sys.exit("Could not parse time {}: check and retry".format(st))

def time2epoch(st):
    str_time = datetime.strptime(st.isoformat(), "%Y-%m-%dT%H:%M:%S")
    str_tuple = str_time.timetuple()
    epoch_time = mktime(str_tuple)
    return epoch_time

def search_payload(item_type,asset_type,geom,start,end,cloud_cover):
    asset_type = f"assets.{asset_type}:download"

    # get images that overlap with our AOI 
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geom
    }

    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": time2utc(start),
        "lte": time2utc(end)
      }
    }

    # only get images which have <10% cloud coverage
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cloud_cover
      }
    }

    asset_filter = {
        "type": "PermissionFilter",
        "config": [asset_type]
    }
    quality = {
        "field_name": "quality_category", 
        "type": "StringInFilter", 
        "config": ["standard"]
    }

    # combine our geo, date, cloud filters
    combined_filter = {
      "type": "AndFilter",
      "config": [geometry_filter, date_range_filter, cloud_cover_filter,asset_filter,quality]
    }

    #item_type = "PSScene4Band"

    # API request object
    search_request = {
      "item_types": [item_type], 
      "filter": combined_filter
    }
    return search_request



#### Search function to return ID list
# This function is the main iterator and looks for all image IDs that matches up with the given search parameters.

def yield_features(url,auth,payload):
    page = requests.post(url, auth=auth, data=json.dumps(payload),headers=headers)
    for feature in page.json()['features']:
        yield feature
    while True:
        url = page.json()['_links']['_next']
        page = requests.get(url, auth=auth)

        for feature in page.json()['features']:
            yield feature

        if page.json()['_links'].get('_next') is None:
            break

def ft_iterate(item_type,asset_type,geom,start,end,cloud_cover,ovp):
    id_master=[]
    ar=[]
    far=[]
    id_ovp = []
    instrument = []
    if ovp is None:
        ovp=0.1 # 10% overlap at least
    search_json = search_payload(item_type,asset_type,geom,start,end,cloud_cover)
    all_features = list(
        yield_features('https://api.planet.com/data/v1/quick-search',
                       HTTPBasicAuth(PLANET_API_KEY, ''), search_json))
    image_ids = [x['id'] for x in all_features]
    aoi_shape = shape(geom)
    for ids in image_ids:
        try:
            id_master.append(ids)
        except Exception as e:
            print(e)
    for feature in all_features:
        s = shape(feature['geometry'])
        # epsgcode = feature["properties"]["epsg_code"] 
        # the updated version does not have epsgcode, so here I set it as 32610 for default.  - KY 20230605
        epsgcode = 32610 # for California zone10
        if aoi_shape.area > s.area:
            intersect = (s).intersection(aoi_shape)
        elif s.area >= aoi_shape.area:
            intersect = (aoi_shape).intersection(s)
        proj_transform = pyproj.Transformer.from_proj(
            pyproj.Proj(4326), pyproj.Proj(epsgcode), always_xy=True # WGS84
        ).transform  # always_xy determines correct coord order
        
        if (
            transform(proj_transform, (aoi_shape)).area
            > transform(proj_transform, s).area
        ):
            if (
                transform(proj_transform, intersect).area
                / transform(proj_transform, s).area
                * 100
            ) >= ovp:
                ar.append(transform(proj_transform, intersect).area / 1000000)
                far.append(transform(proj_transform, s).area / 1000000)
                id_ovp.append(feature['id'])
                instrument.append(feature["properties"]["instrument"])
        elif (
            transform(proj_transform, s).area
            >= transform(proj_transform, aoi_shape).area
        ):
            if (
                transform(proj_transform, intersect).area
                / transform(proj_transform, aoi_shape).area
                * 100
            ) >= ovp:
                ar.append(transform(proj_transform, intersect).area / 1000000)
                far.append(transform(proj_transform, s).area / 1000000)
                id_ovp.append(feature['id'])
                instrument.append(feature["properties"]["instrument"])

    print(f"Total estimated cost to quota: {round(sum(far),3)} sqkm")
    print(f"Total estimated cost to quota if clipped: {round(sum(ar),3)} sqkm")
    # print(ar)

#     print("Remove multiple images for one day") # only one date is needed
#     Because the data quality of different sensor are not the same, here we download all available data

#     dateuse = []
#     iduse = []
#     datelist = [date[0:8] for date in id_ovp]
#     d = {"file": id_ovp, "date": datelist}
#     d = pd.DataFrame(data=d)

#     for x,y in zip(d["file"], d["date"]):
#         if not y in dateuse:
#             dateuse.append(y)
#             iduse.append(x)
#         else:
#             continue

#     id_ovp.sort()
    print(f'Total unique image IDs: {len(list(set(id_ovp)))}')
    # print(f'Total unique image IDs: {list(set(id_master))}') # all available imgs
    print(f'Total used image IDs: {list(set(id_ovp))}') # imgs with ovp > default ovp

    datelist = [d[0:8] for d in id_ovp]
    # return pd.DataFrame(data = {"id": id_ovp, "date": datelist, "instrument": instrument, "estimated area": round(sum(ar),3)})
    return pd.DataFrame(data = {"id": id_ovp, "date": datelist, "instrument": instrument, "estimated area": [round(i,3) for i in ar]})

def order_now(order_payload):
    orders_url = 'https://api.planet.com/compute/ops/orders/v2'
    response = requests.post(orders_url, data=json.dumps(order_payload), auth=(PLANET_API_KEY, ""), headers=headers)
    if response.status_code==202:
        order_id =response.json()['id']
        url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
        feature_check = requests.get(url, auth=(PLANET_API_KEY, ""))
        if feature_check.status_code==200:
            print(f"Submitted a total of {len(feature_check.json()['products'][0]['item_ids'])} image ids: accepted a total of {len(feature_check.json()['products'][0]['item_ids'])} ids")
            print(f"Order URL: https://api.planet.com/compute/ops/orders/v2/{order_id}")
            return f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
    else:
        print(f'Failed with Exception code : {response.status_code}')

def poll_for_success(order_url, num_loops=30):
    count = 0
    while(count < num_loops):
        count += 1
        r = requests.get(order_url, auth=(PLANET_API_KEY, ""))
        response = r.json()
        state = response['state']
        print(f'Order has state {state}')
        end_states = ['success', 'failed', 'partial']
        if state in end_states:
            print(f'Order has state {state}')
            break
        time.sleep(60)


#### Download Block
# Order timings are variable, since we included an email notification for the order, 
# you can wait for the email notification and then run this block. 
# The download block allow you to use the order url to download your files from 
# a specific order if the order has completed either with a state of success to partial


# @retry(stop_max_attempt_number=7)
# def stop_after_7_attempts():
#     print "Stopping after 7 attempts"

@retry(stop_max_attempt_number=7)
def download_results(order_url,folder, overwrite=False):
    r = requests.get(order_url, auth=(PLANET_API_KEY, ""))
    try:
        if r.status_code ==200:
            response = r.json()
            results = response['_links']['results']
            results_urls = [r['location'] for r in results]
            results_names = [r['name'] for r in results]
            print('{} items to download'.format(len(results_urls)))

            for url, name in zip(results_urls, results_names):
                path = pathlib.Path(os.path.join(folder,name))

                if overwrite or not path.exists():
                    print('downloading {} to {}'.format(name, path))
                    r = requests.get(url, allow_redirects=True)
                    # r = requests.get(url, allow_redirects=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    open(path, 'wb').write(r.content)
                else:
                    print('{} already exists, skipping {}'.format(path, name))
        else:
            print(f'Failed with response {r.status_code}')
    except Exception as e:
        print(e)
        print(order_url)
        raise Exception

    r.close()




# Get area to use for clipping and create an order payload
def order_payload(Name_download, ID_imgs, File_geom): 
    with open(File_geom) as f:
        geometry = json.load(f)['features'][0]['geometry']
        
    payload = {
         "name":Name_download, # change order name to whatever you would like (name is not unique)
         "order_type":"partial", # the partial option here allows for an order to complete even if few items fail
         "notifications":{
             "email": True
         },
        "products":[  
            {  
                "item_ids":ID_imgs,#idlist,
                "item_type":"PSScene",#"PSScene4Band",
                "product_bundle": "analytic_sr_udm2"#"analytic_udm2"#""
            }
        ],
        "tools": [
        {
            "clip": {
                "aoi":geometry
            }
        }
        ]
    }
    return payload



