import os

project_folder = 'path_to_code_base'  # Replace with the actual path to your project folder
ROOT_OUTPUT = os.path.join(project_folder, 'output/evaluation')
ROOT_CHECKPOINTS = os.path.join(project_folder, 'model_weights')

data_folder = os.path.join(project_folder, 'toy_data')
ROOT_DATA_SYNTHETIC = os.path.join(data_folder, 'synthetic_data')
ROOT_DATA_LUNG_CT = os.path.join(data_folder, 'NLST')
