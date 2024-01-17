import glob
from get_planet import *
from os.path import exists



# dir_geom = "/Users/kehanyang/Documents/resarch/pc2_meadows/data/Sierra_meadows/geojson_test/"
dir_geom = "/Users/kehanyang/Documents/resarch/pc2_meadows/data/snotel/meadow_selected_extent_geojson_download2/"
dir_download = "/Users/kehanyang/Documents/resarch/pc2_meadows/data/planet/orders/Meadows/"
dir_meadow_images = "/Users/kehanyang/Documents/resarch/pc2_meadows/data/planet/orders/Meadows/"
file_orders_root = "/Users/kehanyang/Documents/resarch/pc2_meadows/data/planet/orders/Orders_URL_Meadows"
flag_search = False
flag_download = True

ID_period = '2017'
file_orders = file_orders_root+ID_period+'.txt'


if flag_search:
    start_time = ID_period + '-01-01T00:00:00'
    # end_time = '2021-12-31T12:00:00'
    end_time = ID_period + '-12-31T12:00:00'
    overlap = 99 # at least with 99% overlap 
    cloud = 0.05 # no more than 5% cloud cover

    fn = glob.glob(dir_geom + "*geojson")
    ID_shp = [id.split("_")[-3] for id in fn]
    df = pd.DataFrame(data = {
        "file": fn, 
        "index": [int(i.split("_")[-3]) for i in fn],
        "ID": [id.split("/")[-1] for id in fn]
        })
    df = df.sort_values("index", ascending = True)

    print(df.head())


    idx = 0 
    if exists(file_orders):
        order_urls = pd.read_csv(file_orders)
    else:
        order_urls = pd.DataFrame(columns = {"index","ID_geom", "order_url"})


    for irow in df.itertuples():
    
    # Search id 
        
        print(irow)
        ID_geom = irow.ID.split(".")[0]+ '_' + ID_period
        print(ID_geom)

        if ID_geom not in order_urls.ID_geom.to_list():

            print('Searching available images ------- ')
            idlist = ft_iterate(item_type='PSScene4Band',
                    asset_type= 'analytic_sr',
                    geom = read_geom(irow.file),#".json"),
                    start = start_time,
                    end = end_time,
                    cloud_cover = cloud, #cloud cover range 0-1 represting 0-100% so 0.5 means max allowed 50% cloud cover
                    ovp = overlap) #% minimum % overlap 0-100

            print(idlist.shape)
            idlist.sort_values("date")
            idlist.to_csv(dir_meadow_images+ID_geom+'.csv')
            
            # print(irow.file)
            payload_info = order_payload(Name_download = ID_geom, ID_imgs = idlist.id.values.tolist(), File_geom = irow.file)
            # print(payload_info)
            print("Pay order:".format(),ID_geom)


            order_url = order_now(payload_info) # error response 400  

            order_urls.loc[idx, "index"] = idx        
            order_urls.loc[idx, "ID_geom"] = ID_geom
            order_urls.loc[idx, "order_url"] = order_url


            # order_urls.append(order_url)  # save all URLs
            order_urls.to_csv(file_orders, index = None)# save all URLs

            
        idx = idx + 1



# after receive the email noticifications
# read order URL from file_orders

if flag_download:
    order_urls_read = pd.read_csv(file_orders)

    for url in order_urls_read.itertuples():
        print(url.order_url)
        # if poll_for_success(url.order_url):
        if os.path.exists(dir_download + url.ID_geom):
            print("Data have been downloaded".format(), dir_download + url.ID_geom)
        else:
            print("start downloading data to".format(), dir_download + url.ID_geom)
            download_results(url.order_url,folder = dir_download + url.ID_geom)



# check the data and download missing data again 
#check whether all data have been downloaded 
# read search csv 
dir_search = '/Users/kehanyang/Documents/resarch/pc2_meadows/data/planet/orders/Meadows/'
fn = glob.glob(dir_search + '*.csv')
id_miss = []
for i in range(0, len(fn)-1):
    data = pd.read_csv(fn[i])
    id = os.path.basename(fn[i]).split('.csv')[0]
    # print(id)
    # print(data[["id",'date','instrument']])
    data['id_three'] = [(i.split("_")[0] + '_' +  i.split("_")[1] + '_' + i.split("_")[2]) for i in data['id']]

    dir_image = dir_search + id
    # print(dir_image)
    fn_img = glob.glob(dir_image + '/**/**/*.tif', recursive = True)
    fn_img_names = [os.path.basename(f) for f in fn_img]
    id_downloaded = [(i.split("_")[0] + '_' +  i.split("_")[1] + '_' + i.split("_")[2]) for i in fn_img_names]

    not_downloaded = data[~data['id_three'].isin(id_downloaded)]

    if len(not_downloaded) > 0:
        print(id)
        print(not_downloaded)
        id_miss.append(id)

id_downloaded
not_downloaded, id_downloaded
print(len(id_miss))
id_miss