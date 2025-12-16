import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from multiprocessing import Pool
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
from netCDF4 import Dataset

from GOESProduct import GOESProduct as GP
from GOESProduct import valid_goes_products

def init_s3_session():
    global s3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def mp_get_goes_day(args):
    return get_goes_day(**args)

def parse_goes_stime(fname:str):
    return datetime.strptime(fname.split("_")[3], "s%Y%j%H%M%S%f")

def get_goes_day(date, products, time_offset_threshold:timedelta):
    """
    Download GOES files for a single day from the AWS bucket. Specify the
    products to acquire as a 2-tuple (goes_product, bands_tods) such that
    goes_product is a valid GOESProduct object, and bands_tods is a list of
    2-tuples (substring, utc_tod). Each bands_tods member describes a single
    file to search for, defined by a combination of a file name substring and
    a UTC time of day (defined as a timedelta) associated with that day.
    """
    downloads = [] ## (product, band, hour, path)
    thresh = time_offset_threshold.seconds
    for prod,bands_tods in products:
        assert prod in valid_goes_products
        hours_bands_times = {}
        for b,t in bands_tods:
            h = t.seconds//3600
            if h not in hours_bands_times.keys():
                hours_bands_times[h] = {b:[]}
            if b not in hours_bands_times[h].keys():
                hours_bands_times[h][b] = [t]
            else:
                hours_bands_times[h][b].append(t)
        bucket = f"noaa-goes{prod.satellite}"
        dstr = date.strftime("%Y/%j")
        for h in hours_bands_times.keys():
            prefix = f"{prod.sensor}-{prod.level}-{prod.scan}/{dstr}/{h:02}"
            response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
                    ).get("Contents", [])
            for b in hours_bands_times[h].keys():
                times,keys = zip(*[
                    (parse_goes_stime(r["Key"]).timestamp(),r["Key"])
                    for r in response if b in r["Key"].replace(prefix,"")
                    ])
                times = np.array(times)
                for t in hours_bands_times[h][b]:
                    dt = np.abs(times-(date+t).timestamp())
                    ix = np.argmin(dt)
                    if dt[ix] < time_offset_threshold.seconds:
                        downloads.append(keys[ix])
    return downloads

if __name__=="__main__":
    sday = datetime(2019,3,15)
    eday = datetime(2019,4,1)
    l1b_bands = [1,2,3,5,6,7,13,15]
    swtimes = [timedelta(hours=t) for t in [12,15,18,21,0]]
    lwtimes = [timedelta(hours=t) for t in [12,15,18,21,0,3,6,9]]
    nworkers = 6

    all_days = []
    cday = sday
    while cday <= eday:
        all_days.append(cday)
        cday = cday + timedelta(days=1)
    pprint(all_days)

    l1b_bands_hours = [
        (f"C{n:02}",t)
        for n in l1b_bands
        for t in [swtimes,lwtimes][n>7]
        ]
    acmc_bands_hours = [("", t) for t in lwtimes]
    products=[
        (GP("16", "ABI", "L2", "ACMC"), acmc_bands_hours),
        (GP("16", "ABI", "L1b", "RadC"), l1b_bands_hours),
        ]
    args = [{
        "date":d,
        "products":products,
        "time_offset_threshold":timedelta(minutes=15)
        } for d in all_days]

    with Pool(nworkers, initializer=init_s3_session) as pool:
        for dl in pool.imap_unordered(mp_get_goes_day, args):
            pprint(dl)
