import os
import glob
import shutil
import pandas as pd
import argparse

def split_dataset(category: str, site: str, ext_path: str = None):
    # Specify the path of the csv file to process
    base_path = 'train_data' if not ext_path else os.path.join(ext_path, 'train_data')
    csv_to_process_path = os.path.join(base_path, f'{site}' , f'{category}_data')
    
    on_dataset_path = os.path.join(base_path, f'{site}',f'{category}_data_on')
    off_dataset_path = os.path.join(base_path, f'{site}',f'{category}_data_off')
    
    # The path of all csv files of the category
    file_path_list = glob.glob(os.path.join(csv_to_process_path, '*', '*', '*', '*', '*'))
    
    # The path of the csv containing the clustering result
    clustered_df_path = glob.glob(os.path.join('src', 'prepare_dataset', 'clustering_result', '*on.csv'))
    print(clustered_df_path)
    
    print("splitting dataset...")
    on_dataset = []
    off_dataset = []
        
    for df_path in clustered_df_path:
        on_dataset_hash = {}
        data_on_df = pd.read_csv(df_path)
        print(data_on_df.shape)
    
        # Split the 'file_name' into parts
        data_on_df[['sensor_id', 'date', 'time']] = data_on_df['file_name'].str.split('_', expand=True).iloc[:, :3]

        # Create a dictionary with machine_type as the key and the relevant data as the value
        machine_type = df_path.split('\\')[-1].split('_')[0]
        on_dataset_hash = {machine_type: list(zip(data_on_df['sensor_id'], data_on_df['date'], data_on_df['time']))}
            
        # Split off, abnormal dataset
        for file_path in file_path_list:
            file_name = os.path.basename(file_path)
            sensor_id_from_file, date_from_file, time_from_file = file_name.split('_')[:3]
            machine_name, machine_type, date_time = file_path.split('\\')[-5:-2]
            
            target_off_dir_path = os.path.join(off_dataset_path, machine_name, machine_type, date_time, sensor_id_from_file)
            target_off_csv_path = os.path.join(target_off_dir_path, file_name)
            
            target_on_dir_path = os.path.join(on_dataset_path, machine_name, machine_type, date_time, sensor_id_from_file)
            target_on_csv_path = os.path.join(target_on_dir_path, file_name)
            
            # Off - dataset
            if machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) not in on_dataset_hash[machine_type]:
                off_dataset.append(file_name)
                os.makedirs(target_off_dir_path, exist_ok=True)
                if os.path.exists(target_off_csv_path):
                    continue
                shutil.copy(file_path, target_off_csv_path)
                
            elif machine_type in on_dataset_hash.keys() and (sensor_id_from_file, date_from_file, time_from_file) in on_dataset_hash[machine_type]:
                on_dataset.append(file_name)
                os.makedirs(target_on_dir_path, exist_ok=True)
                if os.path.exists(target_on_csv_path):
                    continue
                shutil.copy(file_path, target_on_csv_path)

    print(f'Train dataset: {len(on_dataset)}')
    print(f'Off dataset: {len(off_dataset)}')
                                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into On, Off, Evaluate.')
    parser.add_argument('--category', '-c', type=str, default='acc', help='Value for process the specific category of data')
    parser.add_argument('--site', '-s', type=str, default='emsd2_tswh', help='Value for process the specific site of data')
    parser.add_argument('--ext_path', '-p', type=str, default=None, help='Specify the path to download the dataset')
    args = parser.parse_args()
    split_dataset(category=args.category, site=args.site, ext_path=args.ext_path)