# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:33:00 2024

@author: lexi
"""

def clean_colic_data(df, null_cutoff=5, verify=True):
    """
    Function to clean horse colic data.

    Parameters
    ----------
    df : DataFrame
        DataFrame to clean. Assumes data is formatted like Kaggle download with same columns.
    null_cutoff : int, optional
        Number of null values in a row that results in the row being dropped. The default is 5.
    verify : bool, optional
        Whether to print null value counts for cleaned DataFrame. The default is True.

    Returns
    -------
    DataFrame
        Cleaned DataFrame.

    """
    horses_clean = df
    
    # Removing irrelevant columns
    horses_clean = df.drop(['cp_data', 'surgical_lesion', 'lesion_1', 'lesion_2', 'lesion_3'], axis=1)
    
    
    ## Cleaning nasogastric columns
    # Create nasogastric_done column
    horses_clean['nasogastric_done'] = 1 
    no_tube = (horses_clean['nasogastric_tube'].isnull()) & (horses_clean['nasogastric_reflux'].isnull())
    horses_clean.loc[no_tube, 'nasogastric_done'] = 0
    horses_clean.loc[no_tube, 'nasogastric_tube'] = 'n_a'
    horses_clean.loc[no_tube, 'nasogastric_reflux'] = 'n_a'
    
    # Remaining nulls indicate none
    horses_clean['nasogastric_tube'].fillna('none', inplace=True)
    horses_clean['nasogastric_reflux'].fillna('none', inplace=True)
    
    # Remove nasogastric_reflux_ph
    horses_clean = horses_clean.drop('nasogastric_reflux_ph', axis=1)
    
    
    # Remove abdominocentesis columns
    horses_clean = horses_clean.drop(['abdomo_appearance','abdomo_protein'], axis=1)
    
    
    # Row null counts
    rownull_counts = horses_clean.isnull().sum(axis=1).to_frame()
    horses_clean['rownull_counts'] = rownull_counts
    horses_clean = horses_clean[horses_clean['rownull_counts'] < null_cutoff]
    
    
    # Remove duplicates
    horses_clean = horses_clean.drop_duplicates()
    
    
    ## Cleaning categorical columns
    # Fill null 'age' values with 'adult'
    horses_clean['age'].fillna('adult', inplace=True)
    
    # Clean capillary_refill_time values of 3 and fill null values
    horses_clean.loc[horses_clean['capillary_refill_time'] == '3', 'capillary_refill_time'] = 'more_3_sec'
    horses_clean['capillary_refill_time'].fillna('less_3_sec', inplace=True)
    
    # Convert ordered rank categories into integers
    pain_dict = {'alert':0, 'depressed':1, 'mild_pain':2, 'severe_pain':3, 'extreme_pain':4}
    abdis_dict = {'none':0, 'slight':1, 'moderate':2, 'severe':3}

    horses_clean['pain_rank'] = horses_clean['pain'].map(pain_dict)
    horses_clean['abdominal_distention_rank'] = horses_clean['abdominal_distention'].map(abdis_dict)
    
    # Drop original category columns and additional high-null columns
    horses_clean.dropna(subset=['abdominal_distention', 'pain'], inplace=True)
    horses_clean.drop(['abdominal_distention', 'pain', 'abdomen', 'rectal_exam_feces'], axis=1, inplace=True)
    
    # Replace null values of remaining categorical columns with mode
    mode_cols = ['temp_of_extremities','peripheral_pulse','mucous_membrane','peristalsis']
    for col in mode_cols:
        horses_clean.loc[horses_clean[col].isnull(), col] = horses_clean[col].mode().iloc[0]
        
        
    ## Cleaning numerical columns
    horses_clean['rectal_temp'].fillna(horses_clean['rectal_temp'].mean(), inplace=True)
    horses_clean['packed_cell_volume'].fillna(horses_clean['packed_cell_volume'].mean(), inplace=True)

    horses_clean['pulse'].fillna(40, inplace=True)
    horses_clean['respiratory_rate'].fillna(horses_clean['respiratory_rate'].median(), inplace=True) # median is close to mode

    horses_clean['total_protein'].fillna(horses_clean['total_protein'].median(), inplace=True)
    
    
    # Cleanup
    horses_clean.drop('rownull_counts', axis=1, inplace=True)
    
    
    # Verify cleaning
    if verify:
        print('Sum of null values per column:\n', horses_clean.isna().sum(), sep='')
    
    
    return horses_clean


def format_colic_data(filepath):
    """
    Takes colic data from UCI repository and formats it to match data from Kaggle.

    Parameters
    ----------
    filepath : str
        Path to UCI data to be formatted.

    Returns
    -------
    DataFrame
        DataFrame formatted to Kaggle format.

    """
    import pandas as pd
    import numpy as np
    
    # Read data
    col_names = ['surgery', 'age', 'hospital_number', 'rectal_temp', 'pulse',
            'respiratory_rate', 'temp_of_extremities', 'peripheral_pulse',
            'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis',
            'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux',
            'nasogastric_reflux_ph', 'rectal_exam_feces', 'abdomen',
            'packed_cell_volume', 'total_protein', 'abdomo_appearance',
            'abdomo_protein', 'outcome', 'surgical_lesion', 'lesion_1',
            'lesion_2', 'lesion_3', 'cp_data']
    df = pd.read_csv(filepath, delim_whitespace=True, names=col_names)
    
    # Transform '?' to nulls
    df = df.replace('?', np.nan)
    
    # Maps to align imported data to Kaggle column values
    cleaning_maps = {
        'surgery': {'1': 'yes', '2': 'no'},
        'age': {1: 'adult', 2: 'young'},
        'temp_of_extremities': {'1': 'normal', '2': 'warm', '3': 'cool', '4': 'cold'},
        'peripheral_pulse': {'1': 'normal', '2': 'increased', '3': 'reduced', '4': 'absent'},
        'mucous_membrane': {'1': 'normal_pink', '2': 'bright_pink', '3': 'pale_pink', '4': 'pale_cyanotic', '5': 'bright_red', '6': 'dark_cyanotic'},
        'capillary_refill_time': {'1': 'more_3_sec', '2': 'less_3_sec'},
        'pain': {'1': 'alert', '2': 'depressed', '3': 'mild_pain', '4': 'severe_pain', '5': 'extreme_pain'},
        'peristalsis': {'1': 'hypermotile', '2': 'normal', '3': 'hypomotile', '4': 'absent'},
        'abdominal_distention': {'1': 'none', '2': 'slight', '3': 'moderate', '4': 'severe'},
        'nasogastric_tube': {'1': 'none', '2': 'slight', '3': 'significant'},
        'nasogastric_reflux': {'1': 'none', '2': 'more_1_liter', '3': 'less_1_liter'},
        'rectal_exam_feces': {'1': 'normal', '2': 'increased', '3': 'decreased', '4': 'absent'},
        'abdomen': {'1': 'normal', '2': 'other', '3': 'firm', '4': 'distend_small', '5': 'distend_large'},
        'abdomo_appearance': {'1': 'clear', '2': 'cloudy', '3': 'serosanguinous'},
        'outcome': {'1': 'lived', '2': 'died', '3': 'euthanized'},
        'surgical_lesion': {1: 'yes', 2: 'no'},
        'cp_data': {1: 'yes', 2: 'no'}
        }
    
    def map_values(df, cols, map_dict):
        """
        Maps multiple column values of a dataframe to new values using mapping dictionaries.

        Parameters
        ----------
        df : DataFrame
            DataFrame to map columns of.
        cols : list
            List of columns to map new values to.
        map_dict : dict
            Dictionary containing column names as keys and mapping dictionaries as values.

        Returns
        -------
        DataFrame
            The mapped dataframe.

        """
        df = df
        for col in cols:
            if col in map_dict:
                df[col] = df[col].map(map_dict[col])
            else:
                pass
        
        return df
    
    df = map_values(df, col_names, cleaning_maps)
    
    # Transform necessary columns to floats
    float_cols = ['rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph', 'packed_cell_volume', 'total_protein', 'abdomo_protein']
    
    for col in float_cols:
        df[col] = df[col].astype(float)
        
    return df