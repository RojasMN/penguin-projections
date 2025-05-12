import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os # type: ignore

def filter_data(data):
    
    print('FILTERING DATA...')
    
    """
    This function filters the input dataset by removing all rows with missing values in the 'penguin_count' column.
    It then retains only those sites where the available data spans at least 10 years (regardless of data continuity).
    After filtering, the function selects relevant columns and aggregates penguin counts by averaging values
    for the same site, year, count type, and species.
    
    Returns: A cleaned and aggregated DataFrame containing only valid sites and complete penguin count data.
        
    """
    
    sites = list(data['site_id'].unique())
    filtered_sites = []
    removed_sites = []

    for site_id in sites:
        data_site = data[data['site_id'] == site_id].dropna(subset=['penguin_count'])
        
        if data_site.empty:
            removed_sites.append(site_id)
            continue
        
        span = data_site['year'].max() - data_site['year'].min()
        if span >= 10:
            filtered_sites.append(site_id)
        else:
            removed_sites.append(site_id)

    data_filtered = data[data['site_id'].isin(filtered_sites)].dropna(subset=['penguin_count'])
    
    data_filtered = data_filtered[['site_name', 'site_id', 'year', 'penguin_count', 'count_type', 'name_id']]
    data_filtered = data_filtered.groupby(['site_name', 'site_id', 'year', 'count_type', 'name_id'], as_index = False)['penguin_count'].mean()
    
    return data_filtered


def extract_valid_chunks(years, min_span = 10, max_gap = 2):
    
    """
    Identifies continuous segments of years that meet the following criteria:
    
    1. The time span between the first and last year in the segment is at least 'min_span' years.
    2. Gaps between consecutive years within the segment do not exceed 'max_gap' years.
    
    Returns: A list of tuples, where each tuple represents a valid segment as (start_year, end_year).
    
    """
    
    if not years:
        return []

    years = sorted(set(years))
    chunks = []
    current_chunk = [years[0]]

    for y in years[1:]:
        if y - current_chunk[-1] <= max_gap:
            current_chunk.append(y)
        else:
            # Saves the segment if its valid, and continues with the next one
            if current_chunk[-1] - current_chunk[0] >= min_span:
                chunks.append((current_chunk[0], current_chunk[-1]))
            current_chunk = [y]
    
    # Checks last segment 
    if current_chunk[-1] - current_chunk[0] >= min_span:
        chunks.append((current_chunk[0], current_chunk[-1]))
    
    return chunks


def select_good_chunks(data):
    
    print('SELECTING GOOD CHUNKS...')
    
    """
    Segments the dataset into subsets that meet the criteria defined in 'extract_valid_chunks'.

    This function processes each unique site and count type ('nests', 'adults', 'chicks') separately.
    For each combination, it extracts valid year segments that span at least 10 years with a maximum 
    gap of 2 years between consecutive data points.

    Returns: A list of dataframes, each corresponding to a valid time segment for a specific site and count type.
    
    """
    
    types = ['nests', 'adults', 'chicks']
    dataframes = []
    
    for count_type in types:
        data_count = data[(data['count_type'] == count_type)]
        sites = list(data_count['site_id'].unique())
        
        for site in sites:
            data_count_site = data_count[(data_count['site_id'] == site)]
            years = list(data_count_site['year'])
            good_chunks = extract_valid_chunks(years, min_span = 10, max_gap = 2)
        
            for start, end in good_chunks:
                data_chunk = data_count_site[(data_count_site['year'] >= start) & (data_count_site['year'] <= end)]
                dataframes.append(data_chunk)
                
    return dataframes


def interpolate_two_year_gaps(data):
    
    print('INTERPOLATING GAPS (AVERAGE)...')
    
    """
    Interpolates single missing years (2-year gaps) in a time series subset by averaging the values
    before and after the gap.

    For each pair of consecutive data points where the difference in years is exactly 2, this function
    creates a new row at the intermediate year with a 'penguin_count' equal to the average of the two 
    surrounding years. All interpolated rows are then added to the original dataset.

    Returns: A dataframe with interpolated rows added.
        
    """
    
    data = data.sort_values('year').reset_index(drop=True)
    interpolated_rows = []

    for i in range(len(data) - 1):
        y1, y2 = data.loc[i, 'year'], data.loc[i + 1, 'year']
        
        if y2 - y1 == 2:
            
            # Interpolates the gap by taking the average of the previous and following year
            y_interp = y1 + 1
            count_interp = (data.loc[i, 'penguin_count'] + data.loc[i + 1, 'penguin_count']) / 2

            new_row = data.loc[i].copy()
            new_row['year'] = y_interp
            new_row['penguin_count'] = count_interp
            interpolated_rows.append(new_row)

    data_interpolated = pd.concat([data, pd.DataFrame(interpolated_rows)], ignore_index=True)
    data_interpolated = data_interpolated.sort_values('year').reset_index(drop=True)

    return data_interpolated


def interpolate_all_subsets(subsets):
    
    """
    Applies interpolation to all dataframes in a list of subsets.

    This function iterates over a list of dataframes (subsets) and applies the 
    'interpolate_two_year_gaps' function to each dataframe. It then returns a 
    list of dataframes with the interpolated values added for each subset.
        
    """
    
    interpolated_subsets = []
    num_subsets = len(subsets)
    
    for i in range(num_subsets):
        interpolated = interpolate_two_year_gaps(subsets[i])
        interpolated_subsets.append(interpolated)

    return interpolated_subsets


def extract_filename_subset(subset_interp, i):
    
    """
    Generates a filename for saving the subset based on the data it contains.

    This function constructs a filename by using the unique values of 'site_id', 
    'count_type', and 'name_id' from the subset data, along with an index 'i' 
    to ensure uniqueness for each file. The resulting filename is used to 
    save the subset as a .csv file.
    
    """
    site_id = subset_interp['site_id'].unique()[0]
    count_type = subset_interp['count_type'].unique()[0]
    name_id = subset_interp['name_id'].unique()[0]

    filename = f'processed_data_{name_id}/{site_id}_{count_type}_{name_id}_{i}.csv'

    return filename


def save_subsets_csv(subsets_interp):
    
    """
    Saves each subset into a uniquely named .csv file within a folder specific to the species.

    This function iterates through all the subsets of interpolated data, creates a folder 
    for each species (if it doesn't already exist), and saves each subset as a .csv file 
    with a unique filename based on its site ID, count type, and name ID
    
    """
    
    for i, subset in enumerate(subsets_interp):
        
        name_id = subset['name_id'].unique()[0]
        os.makedirs(f'processed_data_{name_id}', exist_ok=True)
        
        filename = extract_filename_subset(subset, i)
        subset.to_csv(filename, index=False)
        