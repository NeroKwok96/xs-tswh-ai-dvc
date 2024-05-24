import boto3
import os
import pandas as pd
import numpy as np
import glob
import argparse
from urllib.parse import urlparse
from dotenv import load_dotenv 
import psycopg2
from datetime import datetime

def download_xs_s3_dataset(category: str, site: str, ext_path: str = None):
    s3_client = boto3.client("s3")
    env_path = os.path.join('.env')
    load_dotenv(env_path)
    xsdb_rds_host = os.environ['xsdb_RDS_HOST']
    xsdb_rds_port = os.environ['xsdb_RDS_PORT']
    xsdb_rds_database = os.environ['xsdb_RDS_DATABASE']
    xsdb_rds_user = os.environ['xsdb_RDS_USER']
    xsdb_rds_password = os.environ['xsdb_RDS_PASSWORD']
    # XS database query
    conn_to_xsdb = psycopg2.connect(
        host=xsdb_rds_host,
        port=int(xsdb_rds_port),
        database=xsdb_rds_database,
        user=xsdb_rds_user,
        password=xsdb_rds_password
    )
    cursor_xsdb = conn_to_xsdb.cursor()
    sensor_history_query = """
    SELECT machine_name, s.node_id, period_from, period_to, location_name
    FROM sensor_history sh
    JOIN sensor_location sl ON sh.sensor_location_id = sl.id
    JOIN sensor s ON s.id = sh.sensor_id
    JOIN machine m ON sl.machine_id = m.id
    JOIN floorplan f ON f.id = m.floorplan_id
    JOIN site ON site.id = f.site_id
    JOIN organization o ON o.id = site.organization_id
    WHERE site.site_id = %s
    order by period_from asc;
    """
    cursor_xsdb.execute(sensor_history_query, (site.split('_')[1],))
    sensor_history_rows = cursor_xsdb.fetchall()
    # Sort sensor_history_rows by period_from in descending order
    sensor_history_rows.sort(key=lambda row: row[2], reverse=True)
    
    for index in range(len(sensor_history_rows)):
        machine_name = sensor_history_rows[index][0].rstrip()
        sensor_id = sensor_history_rows[index][1]
        location_name = sensor_history_rows[index][4]
        period_from = sensor_history_rows[index][2].strftime("%Y%m%d_%H%M%S")
        period_to = sensor_history_rows[index][3]
        
        # Avoid downloading a few duplicate datasets
        if period_from < "2024-01-01" and period_to is None:
            continue
        period_to = "20240518_000000" if period_to is None else sensor_history_rows[index][3].strftime("%Y%m%d_%H%M%S")
        
        # Base on the sensor history, create the directory within the corresponding period
        base_path = 'train_data' if not ext_path else os.path.join(ext_path, 'train_data')
        sensor_location_directory = os.path.join(base_path, site, f'{category}_data', machine_name, location_name, period_from, str(sensor_id))
        if not os.path.exists(sensor_location_directory):
            os.makedirs(sensor_location_directory)
        
        # Download the dataset from S3 by url
        url = f"https://xs-web1-data.s3.ap-southeast-1.amazonaws.com/SpiderWeb/analysed_data/{site}/{sensor_id}/csv/{category}/freq/"
        parsed_url = urlparse(url)
        bucket_name = parsed_url.netloc.split(".")[0]
        s3_key = parsed_url.path.lstrip("/")
        paginator = s3_client.get_paginator('list_objects_v2')
        list_objects_params = {
            'Bucket': bucket_name,
            'Prefix': s3_key
        }
        page_iterator = paginator.paginate(**list_objects_params)
        for page in page_iterator:
            for obj in page.get("Contents", []):
                s3_file_path = obj["Key"]
                s3_file_name = os.path.basename(s3_file_path)
                
                # Check if the file should be downloaded based on the timestamp
                file_date = s3_file_name.split('_')[1]
                file_time = s3_file_name.split('_')[2]
                file_timestamp = f"{file_date}_{file_time}"
                if file_timestamp >= period_from and file_timestamp <= period_to:

                    # Download directory
                    download_dir_path = os.path.join(sensor_location_directory, s3_file_name)

                    # Check if the file already exists
                    if os.path.exists(download_dir_path):
                        print(f"File already exists: {s3_file_name}. Skipping...")
                        continue
                    
                    s3_client.download_file(bucket_name, s3_file_path, download_dir_path)
                    print(f"Downloaded: {s3_file_path}")
                    
def save_to_csv(dataset_dict: dict, category: str):
    for file_name, data in dataset_dict.items():
        machine_name = file_name.split('_')[0]
        df = pd.DataFrame(data, columns=['machine_name' ,'sensor_id', 'file_name', f'oa_{category}_y', f'oa_{category}_x', f'oa_{category}_z', 'extracted_datetime'])
        output_dir = os.path.join(f'oa_{category}_csv', machine_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name + '.csv')
        df.to_csv(output_file, index=False)

def compute_oa(category: str, site: str, ext_path: str = None):
    # Get All the local dataset
    base_path = 'train_data' if not ext_path else os.path.join(ext_path, 'train_data')
    csv_to_process_path = os.path.join(base_path, f'{site}', f'{category}_data')
    file_path_list = glob.glob(os.path.join(csv_to_process_path, '*', '*', '*', '*', '*'))
    dataset_dict = {}
    
    for file_path in file_path_list:
        try:
            data = pd.read_csv(file_path)
            # swap the column in to [freq, vertical, horizontal, axial] format
            if data.shape[1] > 4:
                data = data.drop(columns=[data.columns[-1]])
                data = data[['frequency (Hz)', 'vertical', 'horizontal', 'axial']]
                data.to_csv(file_path, index=False)
            
            file_name = os.path.basename(file_path)
            extracted_datetime_list = file_name.split('_')[1:3]
            extracted_datetime = '_'.join(extracted_datetime_list)
            path_components = file_path.split('\\')
            machine_name = path_components[-5]
            location_name = path_components[-4]
            sensor_id_from_file = path_components[-2]
            output_file_name =(f"{machine_name}_{location_name}")
            
            # Calculate the overall acc
            data = np.square(data.to_numpy()[:, 1:])
            oa_data = np.round((np.sqrt(np.sum(data, axis=0) / 1.5)), decimals=6)
            row_list = [machine_name] + [sensor_id_from_file] + [file_name] + oa_data.tolist() + [extracted_datetime]

            # Hash table
            if output_file_name not in dataset_dict:
                dataset_dict[output_file_name] = []
            dataset_dict[output_file_name].append(row_list)
        except:
            print('Empty file')
            
    save_to_csv(dataset_dict, category)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download the dataset of XS base on the sensor history in xsdb')
    parser.add_argument('--category', '-c', type=str, default='acc', help='Value for download the specific category of data')
    parser.add_argument('--site', '-s', type=str, default='emsd2_tswh', help='Value for download the dataset of related site')
    parser.add_argument('--ext_path', '-p', type=str, default=None, help='Specify the path to download the dataset')
    args = parser.parse_args()
    download_xs_s3_dataset(category=args.category, site=args.site, ext_path=args.ext_path)
    compute_oa(category=args.category, site=args.site, ext_path=args.ext_path)