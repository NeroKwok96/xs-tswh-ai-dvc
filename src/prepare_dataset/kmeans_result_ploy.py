import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px

RHS_data = pd.read_csv('src/prepare_dataset/clustering_result/Motor_data_all.csv')

fig = px.scatter_3d(RHS_data, x='oa_acc_x', y='oa_acc_y', z='oa_acc_z', color='cluster_label',
                    labels={'oa_acc_x': 'oa_acc_x', 'oa_acc_y': 'oa_acc_y', 'oa_acc_z': 'oa_acc_z'}, 
                    title='3D Scatter plot with clusters', hover_data=['file_name', 'sensor_id'])

fig.update_traces(marker=dict(size=3))
fig.show()