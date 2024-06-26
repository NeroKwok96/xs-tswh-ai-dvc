import os
import pandas as pd
import glob
from sklearn.cluster import KMeans
import joblib
import argparse

def read_and_cluster_csv(oa_csv_path_list: list, category: str):
    dataframes = []
    for oa_csv_path in oa_csv_path_list:
        df = pd.read_csv(oa_csv_path)
        dataframes.append(df)
        
    # Concatenate all CSV files of the same machine type
    concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
    concatenated_df = concatenated_df.dropna()
    X = concatenated_df[[f'oa_{category}_y', f'oa_{category}_x', f'oa_{category}_z']]
    kmeans = KMeans(n_clusters=10, random_state=0, n_init=10)
    kmeans.fit(X)
    concatenated_df['cluster_label'] = kmeans.labels_
    return concatenated_df, kmeans

def split_on_off_csv(df: pd.DataFrame, machine_type: str, cluster_result_dir: str, category: str):
    on_df = pd.DataFrame()
    off_df = pd.DataFrame()
    off_validation = 0 
    off_cluster_label = None
    
    for _, off_row in df.iterrows():
        # The dataset is considered to represent the machine is in "Off" status if any axis of the oa < 0.1. 
        # If 100 dataset's cluster label for off dataset are the same, that label should be considered as the label for 'Off'. 
        if (off_row[f'oa_{category}_y'] < 0.1 or off_row[f'oa_{category}_x'] < 0.1 or off_row[f'oa_{category}_z'] < 0.1) and off_validation < 100:
            off_cluster_label = off_row['cluster_label']
            if off_row['cluster_label'] != off_cluster_label:
                raise ValueError('Cluster label changed within the loop, need to check the dataframe')
            off_validation += 1
            
    for _, row in df.iterrows():
        if row['cluster_label'] != off_cluster_label:
            on_df = pd.concat([on_df, row.to_frame().T], ignore_index=True)
        else:
            off_df = pd.concat([off_df, row.to_frame().T], ignore_index=True)
            
    print(f'{machine_type} - off_cluster_label: ', off_cluster_label)
    on_df.to_csv(os.path.join(cluster_result_dir, f'{machine_type}_data_on.csv'), index=False)
    off_df.to_csv(os.path.join(cluster_result_dir, f'{machine_type}_data_off.csv'), index=False)
    df.to_csv(os.path.join(cluster_result_dir, f'{machine_type}_data_all.csv'), index=False)
    
def process_sensor_data(category: str, export: bool):
    kmeans_model_dir = os.path.join('../xs-tswh-vel-spec-ai/aws_lambda/kmeans_models')
    os.makedirs(kmeans_model_dir, exist_ok=True)
    cluster_result_dir = os.path.join('src', 'prepare_dataset', 'clustering_result')
    os.makedirs(cluster_result_dir, exist_ok=True)
    
    # Split dataset into fan and motor respectively with their oa
    oa_csv_to_process_path = os.path.join(f'oa_{category}_csv')
    oa_file_path_list = glob.glob(os.path.join(oa_csv_to_process_path, '*', '*'))
    machine_name_hash = {}
    
    for oa_file_path in oa_file_path_list:
        file_name = os.path.basename(oa_file_path)
        machine_name = file_name.split('_')[1].split('.')[0]
        machine_name_hash[machine_name] = machine_name_hash.get(machine_name, []) + [oa_file_path]
        
    for machine_name, oa_file_path_list in machine_name_hash.items():
        csv_concatenated_df, kmeans_model = read_and_cluster_csv(oa_file_path_list, category)
        split_on_off_csv(csv_concatenated_df, machine_name, cluster_result_dir, category)
        
        if export:
            model_filename = os.path.join(kmeans_model_dir, f'{machine_name}_kmeans_model.pkl')
            joblib.dump(kmeans_model, model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster the dataset by kmeans')
    parser.add_argument('--category', '-c', type=str, default='acc', help='Value for process the specific category of data')
    parser.add_argument("--export", "-e", action="store_true", help="Export the kmeans model or not")
    args = parser.parse_args()
    process_sensor_data(category=args.category, export=args.export)