# DVC for XS dataset
Control the version of datasets of different sites.

## Setup
- Setup Virtual Environment for Python
- Install packages via requirements.txt

## Dataset
### TSWH
```bash
# TSWH - pull velocity datasets (2023-03-01 to 2024-04-28)
dvc pull -r vel_data
```

Download frequency datasets by category (e.g. acc, vel, gE, disp)
```bash
python src/prepare_dataset/download_and_compute.py -c <category> -s <site> -tpt <target_period_to> -p <ext_path>
``` 

Cluster the datasets into On and Off.
```bash
python src/prepare_dataset/kmeans_clustering_for_acc.py -c <category? -e
```

Split On and Off datasets in to their directory
```bash
python src/prepare_dataset/split_dataset_on_off.py -c <category> -s <site> -p <ext_path>
```

