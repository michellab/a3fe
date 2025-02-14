import os
import pickle
import yaml
import sys

# 添加 EnsEquil 模块路径
sys.path.append('/home/roy/software/deve/a3fe/a3fe')

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f"Trying to load module: {module}, name: {name}")
        if module.startswith("EnsEquil"):
            module = module.replace("EnsEquil", "a3fe")
        return super().find_class(module, name)

def update_paths_in_pkl(file_path, old_base_path, new_base_path):
    with open(file_path, 'rb') as f:
        data = CustomUnpickler(f).load()

    def update_path(obj):
        if isinstance(obj, str) and old_base_path in obj:
            return obj.replace(old_base_path, new_base_path)
        elif isinstance(obj, list):
            return [update_path(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: update_path(value) for key, value in obj.items()}
        return obj

    updated_data = update_path(data)

    with open(file_path, 'wb') as f:
        pickle.dump(updated_data, f)

def update_paths_in_yaml(file_path, old_base_path, new_base_path):
    with open(file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    def update_path(obj):
        if isinstance(obj, str) and old_base_path in obj:
            return obj.replace(old_base_path, new_base_path)
        elif isinstance(obj, list):
            return [update_path(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: update_path(value) for key, value in obj.items()}
        return obj

    updated_data = update_path(data)

    with open(file_path, 'w') as f:
        yaml.dump(updated_data, f)

def update_paths_in_files(directory, old_base_path, new_base_path):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pkl'):
                update_paths_in_pkl(file_path, old_base_path, new_base_path)
            elif file.endswith('.yaml') or file.endswith('.yml'):
                update_paths_in_yaml(file_path, old_base_path, new_base_path)

# 更新路径
old_base_path = '/home/finlayclark/software/devel/a3fe_michellab/a3fe/data'
new_base_path = '/home/roy/software/deve/a3fe/a3fe/data'

# 更新 data 目录下的所有文件
data_directory = '/home/roy/software/deve/a3fe/a3fe/data'
update_paths_in_files(data_directory, old_base_path, new_base_path)