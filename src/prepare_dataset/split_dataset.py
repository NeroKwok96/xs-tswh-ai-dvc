import os
import glob
import shutil
import pandas as pd
import argparse
from datetime import datetime, timedelta

def split_dataset(category: str, site: str, ext_path: str = None):
    # Specify the path of the csv file to process
    base_path = 'train_data' if not ext_path else os.path.join(ext_path, 'train_data')
    csv_to_process_path = os.path.join(base_path, f'{site}' , f'{category}_data')
    
    on_dataset_path = os.path.join(base_path, f'{site}',f'{category}_data_on')
    off_dataset_path = os.path.join(base_path, f'{site}',f'{category}_data_off')
    evaluate_dataset_path = os.path.join(base_path, f'{site}', f'{category}_data_evaluate')
    
    # The path of all csv files of the category
    file_path_list = glob.glob(os.path.join(csv_to_process_path, '*', '*', '*', '*', '*'))
    
    # The path of the csv containing the clustering result
    clustered_df_path = glob.glob(os.path.join('src', 'on_off_clustering', 'clustering_result', '*'))
    
    bearing_fault_case = [
        {
            "machine_name": "6B ISO Rm 1 Fan no.2",
            "machine_type": "Motor",
            "date_range": ("20231114", "20231120")
        },
        {
            "machine_name": "6A ISO Rm 1 Fan no.1",
            "machine_type": "Motor",
            "date_range": ("20230831", "20230906")
        },
        {
            "machine_name": "7A ISO Rm 2 Fan no.2",
            "machine_type": "Motor",
            "date_range": ("20230814", "20230821")
        },
        {
            "machine_name": "7A ISO Rm 2 Fan no.2",
            "machine_type": "Motor",
            "date_range": ("20231224", "20231228")
        },
        {
            "machine_name": "6B ISO Rm 2 Fan no.1",
            "machine_type": "Motor",
            "date_range": ("20240117", "20240202")
        },
        {
            "machine_name": "5A ISO Rm 2 Fan no.1",
            "machine_type": "Motor",
            "date_range": ("20231201", "20231231")
        }
    ]
    print("splitting dataset...")
    abnormal_dataset = []
    normal_dataset = []
    on_dataset = []
    off_dataset = []
    
    for df_path in clustered_df_path:
        if df_path.endswith('on.csv'):
            on_dataset_hash = {}
            data_on_df = pd.read_csv(df_path)
            
            # Cache the sensor id, date, time of on dataset
            for _, row in data_on_df.iterrows():
                machine_type = df_path.split('\\')[-1].split('_')[0]
                on_data_file_name = row['file_name']
                on_data_sensor_id, on_data_date, on_data_time = on_data_file_name.split('_')[:3]
                on_dataset_hash[machine_type] = on_dataset_hash.get(machine_type, [])
                on_dataset_hash[machine_type].append((on_data_sensor_id, on_data_date, on_data_time))
                
            # Split off, abnormal dataset
            for file_path in file_path_list:
                file_name = os.path.basename(file_path)
                sensor_id_from_file, date_from_file, time_from_file = file_name.split('_')[:3]
                machine_name, machine_type, date_time = file_path.split('\\')[-5:-2]
                
                target_off_dir_path = os.path.join(off_dataset_path, machine_name, machine_type, date_time, sensor_id_from_file)
                target_off_csv_path = os.path.join(target_off_dir_path, file_name)
                
                target_evaluate_abnormal_dir_path = os.path.join(evaluate_dataset_path, 'abnormal', machine_name, machine_type, date_time, sensor_id_from_file)
                target_evaluate_abnormal_csv_path = os.path.join(target_evaluate_abnormal_dir_path, file_name)
                
                # Off - dataset
                if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) not in on_dataset_hash[machine_type]:
                    off_dataset.append(file_name)
                    os.makedirs(target_off_dir_path, exist_ok=True)
                    if os.path.exists(target_off_csv_path):
                        continue
                    shutil.copy(file_path, target_off_csv_path)
                
                # On - dataset
                if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) in on_dataset_hash[machine_type]:
                    
                    for machine in bearing_fault_case:
                        
                        # dataset 10 days before and after the fault case start date and end date will be assumed as abnormal dataset
                        abnormal_start_date = (datetime.strptime(machine['date_range'][0], "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
                        abnormal_end_date = (datetime.strptime(machine['date_range'][1], "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
                        
                        # Evaluate - Abnormal dataset
                        if date_from_file >= abnormal_start_date and date_from_file <= abnormal_end_date and machine_name == machine['machine_name'] and machine_type == machine['machine_type']:
                            abnormal_dataset.append(file_name)
                            os.makedirs(target_evaluate_abnormal_dir_path, exist_ok=True)
                            if os.path.exists(target_evaluate_abnormal_csv_path):
                                continue
                            shutil.copy(file_path, target_evaluate_abnormal_csv_path)
                            
            # Split On, Normal dataset
            for file_path in file_path_list:
                file_name = os.path.basename(file_path)
                sensor_id_from_file, date_from_file, time_from_file = file_name.split('_')[:3]
                machine_name, machine_type, date_time = file_path.split('\\')[-5:-2]
                
                target_evaluate_normal_dir_path = os.path.join(evaluate_dataset_path, 'normal', machine_name, machine_type, date_time, sensor_id_from_file)
                target_evaluate_normal_csv_path = os.path.join(target_evaluate_normal_dir_path, file_name)
                
                target_on_dir_path = os.path.join(on_dataset_path, machine_name, machine_type, date_time, sensor_id_from_file)
                target_on_csv_path = os.path.join(target_on_dir_path, file_name)

                if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) in on_dataset_hash[machine_type]:
                    
                    # Evaluate - Normal dataset
                    if date_from_file >= '20231101' and file_name not in abnormal_dataset:
                        normal_dataset.append(file_name)
                        os.makedirs(target_evaluate_normal_dir_path, exist_ok=True)
                        if os.path.exists(target_evaluate_normal_csv_path):
                            continue
                        shutil.copy(file_path, target_evaluate_normal_csv_path)
                    
                    # On - Training dataset
                    elif date_from_file < '20231101' and file_name not in abnormal_dataset:
                        on_dataset.append(file_name)
                        os.makedirs(target_on_dir_path, exist_ok=True)
                        if os.path.exists(target_on_csv_path):
                            continue
                        shutil.copy(file_path, target_on_csv_path)

    print(f'Normal dataset: {len(normal_dataset)}')
    print(f'Abnormal dataset: {len(abnormal_dataset)}')
    print(f'Train dataset: {len(on_dataset)}')
    print(f'Off dataset: {len(off_dataset)}')
                                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into On, Off, Evaluate.')
    parser.add_argument('--category', '-c', type=str, default='acc', help='Value for process the specific category of data')
    parser.add_argument('--site', '-s', type=str, default='emsd2_tswh', help='Value for process the specific site of data')
    parser.add_argument('--path', '-p', type=str, default=None, help='Specify the path to download the dataset')
    args = parser.parse_args()
    split_dataset(category=args.category, site=args.site, ext_path=args.ext_path)