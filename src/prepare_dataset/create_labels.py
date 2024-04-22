import os
import glob
import pandas as pd
import argparse
from datetime import datetime, timedelta
from fault_case import bearing_fault

def split_dataset(category: str, site: str, ext_path: str = None):
    
    # Specify the path of the csv files to process
    base_path = 'train_data' if not ext_path else os.path.join(ext_path, 'train_data')
    csv_to_process_path = os.path.join(base_path, f'{site}' , f'{category}_data')
    file_path_list = glob.glob(os.path.join(csv_to_process_path, '*', '*', '*', '*', '*'))
    
    # Path of clustered result with only "on" dataset
    clustered_df_path = glob.glob(os.path.join('src', 'on_off_clustering', 'clustering_result', '*on*.csv'))
    
    # Import data of the fault case
    bearing_fault_case = bearing_fault.bearing_fault_case
    
    abnormal_dataset = []
    abnormal_df = []
    normal_df = []
    on_df = []
    off_df = []
    print(off_df)
    # result = pd.DataFrame(columns=["machine_name", "machine_type", "sensor_id", "file_prefix", "category", "csv_type", "classification"])
    
    for df_path in clustered_df_path:
        on_dataset_hash = {}
        data_on_df = pd.read_csv(df_path)
        
        # Split the 'file_name' into parts
        data_on_df[['sensor_id', 'date', 'time']] = data_on_df['file_name'].str.split('_', expand=True).iloc[:, :3]

        # Get the machine type from the df_path
        machine_type = df_path.split('\\')[-1].split('_')[0]

        # Create a dictionary with machine_type as the key and the relevant data as the value
        on_dataset_hash = {machine_type: list(zip(data_on_df['sensor_id'], data_on_df['date'], data_on_df['time']))}
            
        # Split off, abnormal dataset
        for file_path in file_path_list:
            file_name = os.path.basename(file_path)
            sensor_id_from_file, date_from_file, time_from_file = file_name.split('_')[:3]
            machine_name, machine_type, date_time = file_path.split('\\')[-5:-2]
            file_prefix = "_".join(file_name.split('_')[:3])
            
            # Off - dataset
            if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) not in on_dataset_hash[machine_type]:
                
                off_df.append(pd.DataFrame({
                    "machine_name": [machine_name], 
                    "machine_type": [machine_type], 
                    "sensor_id": [sensor_id_from_file], 
                    "file_prefix": [file_prefix], 
                    "category": [category], 
                    "csv_type": ["freq"], 
                    "classification": ["off"]
                    }))
            
            # Evaluate - Abnormal dataset
            if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) in on_dataset_hash[machine_type]:
                
                for machine in bearing_fault_case:
                    
                    # dataset 10 days before and after the fault case start date and end date will be assumed as abnormal dataset
                    abnormal_start_date = (datetime.strptime(machine['date_range'][0], "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
                    abnormal_end_date = (datetime.strptime(machine['date_range'][1], "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
                    
                    # Evaluate - Abnormal dataset
                    if date_from_file >= abnormal_start_date and date_from_file <= abnormal_end_date and machine_name == machine['machine_name'] and machine_type == machine['machine_type']:
                        abnormal_df.append(pd.DataFrame({
                            "machine_name": [machine_name],
                            "machine_type": [machine_type],
                            "sensor_id": [sensor_id_from_file],
                            "file_prefix": [file_prefix],
                            "category": [category],
                            "csv_type": ["freq"],
                            "classification": ["abnormal"]
                        }))
                        abnormal_dataset.append(file_name)
                        
        # Split On, Normal dataset
        for file_path in file_path_list:
            file_name = os.path.basename(file_path)
            sensor_id_from_file, date_from_file, time_from_file = file_name.split('_')[:3]
            machine_name, machine_type, date_time = file_path.split('\\')[-5:-2]
            file_prefix = "_".join(file_name.split('_')[:3])

            if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) in on_dataset_hash[machine_type]:
                
                # Evaluate - Normal dataset
                if date_from_file >= '20231101' and file_name not in abnormal_dataset:
                    normal_df.append(pd.DataFrame({
                        "machine_name": [machine_name],
                        "machine_type": [machine_type],
                        "sensor_id": [sensor_id_from_file],
                        "file_prefix": [file_prefix],
                        "category": [category],
                        "csv_type": ["freq"],
                        "classification": ["normal"]
                    }))
                
                # On - Training dataset
                elif date_from_file < '20231101' and file_name not in abnormal_dataset:
                    on_df.append(pd.DataFrame({
                        "machine_name": [machine_name],
                        "machine_type": [machine_type],
                        "sensor_id": [sensor_id_from_file],
                        "file_prefix": [file_prefix],
                        "category": [category],
                        "csv_type": ["freq"],
                        "classification": ["on"]
                    }))
    
    print(f'Normal df: {len(normal_df)}')
    print(f'Abnormal df: {len(abnormal_df)}')
    print(f'Train df: {len(on_df)}')
    print(f'Off df: {len(off_df)}')
    print(f'Total df: {len(normal_df) + len(abnormal_df) + len(on_df) + len(off_df)}')
    
    concat_on_df = pd.concat(on_df, ignore_index=True)
    concat_off_df = pd.concat(off_df, ignore_index=True)
    concat_abnormal_df = pd.concat(abnormal_df, ignore_index=True)
    concat_normal_df = pd.concat(normal_df, ignore_index=True)
    result = pd.concat([concat_on_df, concat_off_df, concat_abnormal_df, concat_normal_df], ignore_index=True)
    result.to_csv("test_result.csv", index=False)
                                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into On, Off, Evaluate.')
    parser.add_argument('--category', '-c', type=str, default='acc', help='Value for process the specific category of data')
    parser.add_argument('--site', '-s', type=str, default='emsd2_tswh', help='Value for process the specific site of data')
    parser.add_argument('--ext_path', '-p', type=str, default=None, help='Specify the path to download the dataset')
    args = parser.parse_args()
    split_dataset(category=args.category, site=args.site, ext_path=args.ext_path)