import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from shutil import make_archive
np.random.seed(5122025)

split_sizes = [0.03*0.01,1*0.01]

def generate_balance_dataset_according_to_specific_categories(dataframe: pd.DataFrame,
                                                              criteria: list[pd.Series],
                                                              split_size: float)-> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    dataframe: data to split
    criteria: series that are expected to be balanced in the resulting sets
    split_size: proportion in the first resulting set (the second is composed of the other samples) 
    '''
    if not 0 < split_size < 1:
        raise ValueError("split_size must be between 0 and 1")
    
    codes = ["!".join(str(c) for c in row) for row in zip(*criteria)]
    codes = pd.Series(codes, index=dataframe.index)  
    idx = dataframe.index.to_numpy()
 
    unique_values, inverse, res_counts = np.unique(codes, return_counts=True, return_inverse=True)

    idx_per_value = [[] for _ in unique_values]

    for i,a in enumerate(inverse):
        idx_per_value[a].append(i)
    
    idx_split = []
    
    for j in tqdm(range(len(unique_values))):
        available_idx_rest,res_count = idx_per_value[j],res_counts[j]
        res_count = res_count*split_size
        res_count_int = np.floor(res_count)
        pick_count = int(np.random.binomial(1,res_count-res_count_int)+res_count_int)
        available_idx = idx[available_idx_rest]
        np.random.shuffle(available_idx)
        idx_split.append(available_idx[:pick_count])

    idx_split = np.concatenate(idx_split)
    
    take_split = pd.Series(False,codes.index)
    
    take_split.loc[idx_split] = True
    
    training_set = dataframe[take_split]
    
    testing_set = dataframe[~take_split]
    
    return training_set, testing_set


df_individual = pd.read_csv("Generated_Data/full_dataset_Individual.csv", sep=";", low_memory=False)

criteria_individual = [df_individual["Sex"],df_individual["Age"]//5,df_individual["County"]]

for size in split_sizes:
    name_size = str(100*size).replace(".","_") if (100*size)!=np.round(10**size) else str(int(100*size))
    dir = f"Generated_Data/datasets_Individual_{name_size}"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    training_set, testing_set = generate_balance_dataset_according_to_specific_categories(df_individual, criteria_individual, size)
    training_set.to_csv(f"{dir}/training_dataset_Individual_{name_size}.csv", sep=";", index=False)
    testing_set.to_csv(f"{dir}/testing_dataset_Individual_{name_size}.csv", sep=";", index=False)
#     make_archive(dir,'zip', "Generated_Data")


df_household = pd.read_csv("Generated_Data/full_dataset_Household.csv", sep=";", low_memory=False)
df_household_group_by = df_household.groupby("HouseholdID")
criteria_household = [df_household_group_by.size(),df_household_group_by["County"].first()]

household_id_serie = pd.Series(index=criteria_household[0].index.to_numpy())

for size in split_sizes:
    name_size = str(100*size).replace(".","_") if (100*size)!=np.round(10**size) else str(int(100*size))
    dir = f"Generated_Data/datasets_Household_{name_size}"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    training_set_household_id, testing_set_household_id = generate_balance_dataset_according_to_specific_categories(household_id_serie, criteria_household, size)
    
    mask = (df_household["HouseholdID"].isin(training_set_household_id.index.to_numpy()))
    
    training_set = df_household[mask]
    testing_set = df_household[~mask]
    
    training_set.to_csv(f"{dir}/training_dataset_Household_{name_size}.csv", sep=";", index=False)
    testing_set.to_csv(f"{dir}/testing_dataset_Household_{name_size}.csv", sep=";", index=False)
    # make_archive(dir,'zip', "Generated_Data")
