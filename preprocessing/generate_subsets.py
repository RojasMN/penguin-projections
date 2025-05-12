import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os # type: ignore
import processing_tools as ptools # type: ignore

# DATASETS
path_adelia = r'C:\Users\marce\Escritorio\Proyectos\projections_pys\data\query_adelia.csv'
path_barbijo = r'C:\Users\marce\Escritorio\Proyectos\projections_pys\data\query_barbijo.csv'
path_papua = r'C:\Users\marce\Escritorio\Proyectos\projections_pys\data\query_papua.csv'
paths = [path_adelia, path_barbijo, path_papua]
sp_ids = ['adelie', 'chinstrap', 'gentoo']

def full_pipeline(data, species):
    
    """
    This function produces several preprocessed .csv files. 
    More detail about the preprocessing can be found on 'processing_tools.py'
    
    """
    
    name_map = {
    'adelie penguin': 'adelie',
    'chinstrap penguin': 'chinstrap',
    'macaroni penguin': 'macaroni',
    'gentoo penguin': 'gentoo',
    'emperor penguin': 'emperor',
    'king penguin': 'king'
    }

    data['name_id'] = data['common_name'].map(name_map)
    data_sp = data[(data['name_id'] == species)]
    data_filtered = ptools.filter_data(data_sp)
    subsets = ptools.select_good_chunks(data_filtered)
    subsets_interp = ptools.interpolate_all_subsets(subsets)

    ptools.save_subsets_csv(subsets_interp)
    

def main():
    for path, sp_id in zip(paths, sp_ids):
        data = pd.read_csv(path)
        full_pipeline(data, sp_id)
        
main()