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



def process_unseen_values_training_individual(df_train, df_test):
    df_test = process_unseen_non_geographical_values(df_train, df_test)
    df_test = process_unseen_geographical_values_training(df_train, df_test)
    return df_test

def process_unseen_values_training_household(df_train, df_test):
    df_test_2 = process_unseen_non_geographical_values(df_train, df_test)
    df_test_by_household = df_test_2.groupby("HouseholdID")[["Department","County", "City", "TRIRIS", "IRIS"]].first()
    df_test_by_household_2 = process_unseen_geographical_values_training(df_train, df_test_by_household.copy())
    
    geo_attributes = ["Department","County", "City", "TRIRIS", "IRIS"]
    
    for attribute in geo_attributes:
        df_test_2.loc[:,attribute] = df_test_2["HouseholdID"].map(dict(zip(df_test_by_household_2.index, df_test_by_household_2[attribute])))
    return df_test_2

def process_unseen_non_geographical_values(df_train, df_test):
    """
    Remove test-set rows containing unseen non-geographical categorical values.

    This function scans all non-geographical columns (assumed to be all columns except
    the last five, which represent hierarchical geographic data) and identifies any
    categories that appear in the test dataset but do not appear in the training dataset.
    Rows in the test set containing these unseen categories are filtered out entirely.

    Steps performed:
    1. For each non-geographical column:
        - Compute normalized value counts for training and test datasets.
        - Identify categories that appear in the test set but are missing from the
          training set.
    2. For each column, flag test rows containing such unseen categories.
    3. Combine all flags to remove any test row that contains at least one unseen value
       across any non-geographical column.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing reference categorical values.
    df_test : pandas.DataFrame
        Test dataset potentially containing unseen non-geographical categories.

    Returns
    -------
    pandas.DataFrame
        Filtered version of `df_test`, where rows containing unseen non-geographical
        values have been removed.

    Notes
    -----
    - Only non-geographical columns (all except the last 5) are processed.
    - The function performs *row-level filtering* rather than imputation.
    - If a row contains even one unseen value in any non-geographical column, it is
      removed from the returned DataFrame.
    - The original `df_train` is not modified.
    """
    keep_index = pd.Series(1, index=df_test.index)
    for col in df_train.columns[:-6]:
        df = pd.DataFrame([df_train[col].value_counts(normalize=True),df_test[col].value_counts(normalize=True)]).transpose()
        df.columns=["Training","Testing"]
        if (np.sum(df["Training"].isna())>0):
            keep_index = keep_index*(1-df_test[col].isin(df.index[df["Training"].isna()])).astype(int)
    return df_test[keep_index.astype(bool)]

def process_unseen_geographical_values_training(df_train, df_test):
    """
    Handle unseen categorical values in hierarchical geographic features by imputing
    missing categories in the test dataset based on their parent-level distributions.

    This function assumes a hierarchical structure between geographic attributes:
    Department → County → City → TRIRIS → IRIS.
    When a category appears in the test set but not in the training set at a given
    geographic level, it is considered "unseen." These unseen categories are replaced
    using the observed distribution of categories within the same parent geographic area.

    Steps performed:
    1. For each geographic level, identify categories present in `df_test` but missing
       from `df_train`.
    2. For each missing category, determine its parent area (e.g., a missing City’s
       County, or a missing IRIS’s TRIRIS).
    3. Within that parent area, compute the normalized distribution of known categories.
    4. Randomly reassign the unseen values in the test set according to this distribution.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing hierarchical geographic attributes.
    df_test : pandas.DataFrame
        Test dataset where unseen categories may appear.

    Returns
    -------
    df_train : pandas.DataFrame
        Unmodified training dataset (returned for convenience).
    df_test : pandas.DataFrame
        Modified test dataset with unseen categories replaced using parent-level
        probability distributions.

    Notes
    -----
    - Random imputation is performed with `numpy.random.choice`, using probabilities
      derived from the observed frequencies in the same parent area.
    - Only categories missing from the training data are modified; all others remain
      unchanged.
    - Parent areas with no valid distribution are skipped without modification.
    """
    geo_attributes = ["Department","County", "City", "TRIRIS", "IRIS"]
    for geo_previous, geo_current in zip(geo_attributes[:-1], geo_attributes[1:]):
        # Calculate normalized value counts for training and testing data
        train_counts = df_train[geo_current].value_counts(normalize=True)
        test_counts = df_test[geo_current].value_counts(normalize=True)

        # Create a DataFrame to compare training and testing distributions
        df = pd.DataFrame({"Training": train_counts, "Testing": test_counts}).transpose()

        # Identify categories in the test set that are missing from the training set
        missing_values = df.loc["Training"].isna()
        missing_indices = df.loc["Training"][missing_values].index

        if not missing_indices.empty:
            # Boolean mask for rows in the test set with missing categories
            index_to_change = df_test[geo_current].isin(missing_indices)

            # Get unique parent areas for rows with missing categories
            modalities_geo_previous_at_risk = df_test[index_to_change][geo_previous].unique()

            # Iterate over each parent area with missing categories
            for val in modalities_geo_previous_at_risk:
                # Boolean mask for rows in the current parent area
                idx_modalities_geo_previous_at_risk = (df_test[geo_previous] == val)

                # Split into rows to change (missing categories) and rows for probability calculation
                df_test_previous_area_for_proba = df_test[(idx_modalities_geo_previous_at_risk) & (~index_to_change)]
                idx_to_change_val = (idx_modalities_geo_previous_at_risk) & (index_to_change)

                # Calculate the distribution of known categories in the parent area
                if not df_test_previous_area_for_proba.empty:
                    repartition = df_test_previous_area_for_proba[geo_current].value_counts(normalize=True)
                    idx = repartition.index
                    proba = repartition.values

                    # Randomly assign known categories to missing values, based on the parent area's distribution
                    res = np.random.choice(idx, p=proba, size=np.sum(idx_to_change_val))
                    df_test.loc[idx_to_change_val, geo_current] = res
                else:
                    # Raise an error if no valid distribution is found for the parent area
                    raise ImplementationError("No valid distribution found for parent area.")
    return df_test


df_individual = pd.read_csv("Generated_Data/full_dataset_Individual.csv", sep=";", low_memory=False)

criteria_individual = [df_individual["Sex"],df_individual["Age"]//5,df_individual["County"]]

for size in split_sizes:
    name_size = str(100*size).replace(".","_") if (100*size)!=np.round(10**size) else str(int(100*size))
    dir = f"Generated_Data/datasets_Individual_{name_size}"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    training_set, testing_set = generate_balance_dataset_according_to_specific_categories(df_individual, criteria_individual, size)
    testing_set = process_unseen_values_training_individual(training_set, testing_set)
    training_set.to_csv(f"{dir}/training_dataset_Individual_{name_size}.csv", sep=";", index=False)
    testing_set.to_csv(f"{dir}/testing_dataset_Individual_{name_size}.csv", sep=";", index=False)
# #     make_archive(dir,'zip', "Generated_Data")


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
    
    testing_set = process_unseen_values_training_household(training_set, testing_set)
    training_set.to_csv(f"{dir}/training_dataset_Household_{name_size}.csv", sep=";", index=False)
    testing_set.to_csv(f"{dir}/testing_dataset_Household_{name_size}.csv", sep=";", index=False)
    # make_archive(dir,'zip', "Generated_Data")
