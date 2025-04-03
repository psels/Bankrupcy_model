import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from datetime import datetime
import numpy as np
from sklearn.utils import resample
from sklearn import impute
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from datetime import datetime
import pgeocode
from geopy.geocoders import Nominatim
import re

nomi = pgeocode.Nominatim('fr')
geolocator = Nominatim(user_agent="geoapi", timeout=10)

raw_dataset = pd.read_csv("raw_data/dataset 1.00.csv")
nace_industry = pd.read_csv('raw_data/nace_industry.csv', delimiter=';')

categorielle = ['diffusionInsee', 'typePersonne','diffusionCommerciale', 'succursaleOuFiliale',
    'formeExerciceActivitePrincipale', 'societeEtrangere',
    'formeJuridique_1', 'microEntreprise', 'etablieEnFrance',
    'salarieEnFrance', 'relieeEntrepriseAgricole', 'entrepriseAgricole',
    'eirl', "sectionName", "nace_letter"]
other = ['dateCreation', 'nace', 'code_postal']

def get_nace_mapping():
    """Gets the mapping of possible nace codes"""

    df_nace_mapping = pd.read_excel("raw_data/nace_mapping.xlsx")

    dico_nace_mapping = {}

    dico_nace_mapping["unknown"] = "unknown"

    for i, rows in df_nace_mapping.iterrows():
        dico_nace_mapping[str(rows["Four"])] = rows["Letter"]
        dico_nace_mapping[str(rows["Three"])] = rows["Letter"]
        dico_nace_mapping[str(rows["Two"])] = rows["Letter"]

    return dico_nace_mapping

def preprocessing_nace(df):
    """Standardize naces and only keep valid ones and adds industry labels"""

    #standardize
    result_df = df.copy()

    result_df['nace_letter'] = result_df['nace'].str.extract(r'([A-Za-z]+)')[0]
    result_df['nace_four'] = result_df['nace'].str.replace(r'[^0-9]', '', regex=True)

    result_df['nace_four'] = result_df['nace_four'].apply(lambda x: "#" + str(x).ljust(4, '0') if len(str(x)) < 4 else "#" + x)
    result_df['nace_three'] = result_df['nace_four'].str[:4]
    result_df['nace_two'] = result_df['nace_four'].str[:3]

    #only keep valid naces
    dico_nace_mapping = get_nace_mapping()

    result_df["nace_four"] = np.where(result_df["nace_four"].isin(list(dico_nace_mapping.keys())), result_df["nace_four"], "unknown")
    result_df["nace_three"] = np.where(result_df["nace_three"].isin(dico_nace_mapping), result_df["nace_three"], "unknown")
    result_df["nace_two"] = np.where(result_df["nace_two"].isin(dico_nace_mapping), result_df["nace_two"], "unknown")

    result_df.loc[:, "nace_letter"] = result_df.apply(lambda x: dico_nace_mapping.get(x["nace_two"], x["nace_letter"]), axis=1)

    #adds industry labels
    result_df = result_df.merge(right=nace_industry, how="left", left_on='nace_letter', right_on='Letter')
    result_df.drop(columns=["Letter"], axis=1, inplace=True)
    result_df.drop(columns=['nace'], axis=1, inplace=True)
    result_df.rename(columns={"Section Name": "sectionName"}, inplace=True)

    return result_df

def preprocessing_forme_juridique(dataset):

    dataset['formeJuridique_1'] = dataset['formeJuridique_1'].fillna("other")
    dataset['formeJuridique_1'] = dataset['formeJuridique_1'].replace("nan", "other")

    values = dataset['formeJuridique_1'].value_counts()

    dataset['formeJuridique_1'] = dataset['formeJuridique_1'].apply(lambda x: "other" if values[x]<=100 else str(x))

    return dataset

def format_date_creation(X_train, X_test):
    """apply date creation transformation"""

    #X_train["dateCreation"] = X_train["dateCreation"].apply(lambda x: pd.NA if x == "nan" else int(str(x)[:4]))
    X_train["dateCreation"] = X_train["dateCreation"].apply(lambda x: pd.NA if (pd.isna(x) or str(x).lower() == "nan") else int(str(x)[:4]))

    mean_x_train = X_train["dateCreation"].mean(skipna=True)
    X_train["dateCreation"] = X_train["dateCreation"].apply(lambda x: mean_x_train if (pd.isna(x) or str(x).lower() == "nan") else x)

    X_test["dateCreation"] = X_test["dateCreation"].apply(lambda x: pd.NA if (pd.isna(x) or str(x).lower() == "nan") else int(str(x)[:4]))
    X_test["dateCreation"] = X_test["dateCreation"].apply(lambda x: mean_x_train if (pd.isna(x) or str(x).lower() == "nan") else x)

    return X_train, X_test

def create_redistributed_dataset(X, y, desired_default_ratio):
    """redistributes dataset, if the desired default ratio is 0.25, it will have 25% of defaults in the new dataset"""

    # Store original column order
    original_cols = X.columns.tolist()

    # Ensure y is a Series with a name
    if y.name is None:
        y = y.rename('is_defaulted')
    target_col = y.name

    # Create a single dataset with a simple index
    dataset = pd.concat([X.reset_index(drop=True),
                         y.reset_index(drop=True)], axis=1)

    # Split by class
    dataset_full_class_0 = dataset[dataset[target_col] == 0]
    dataset_full_class_1 = dataset[dataset[target_col] == 1]

    # Calculate the size for the majority class
    multiplier = (1/desired_default_ratio) - 1
    n_samples = int(multiplier*len(dataset_full_class_1))

    # Undersample the majority class
    data_class_0_undersampled = resample(
        dataset_full_class_0,
        replace=False,
        n_samples=n_samples,
        random_state=42
    )

    # Combine and shuffle, ensuring index is reset
    dataset_resampled = pd.concat([data_class_0_undersampled, dataset_full_class_1],
                                  ignore_index=True)
    dataset_resampled = dataset_resampled.sample(frac=1, random_state=42)

    # Split back to X and y
    y_new = dataset_resampled[target_col]
    X_new = dataset_resampled.drop(target_col, axis=1)

    # Ensure X has the same columns in the same order as the original
    X_new = X_new[original_cols]

    return X_new, y_new

def separate_x_y(dataset):
    """gets X (features) and y (target)"""

    X = dataset.drop('is_defaulted', axis=1)
    y = dataset['is_defaulted']

    return X, y

def remove_useless_columns(dataset):
    """removes not used columns, should be removed in the sql directly later"""
    dataset = dataset.drop(["lastUpdate", "siren", "id", "formeJuridique"], axis=1) #formetJuridique1 est légèrement + complet

    return dataset

def get_binary_columns(dataset):
    """get lists of binary column names"""

    binary_columns = [column for column in categorielle if dataset[column].nunique()==2]

    return binary_columns

def get_numerical_columns(dataset):
    """get lists of numerical column names"""

    print(dataset.dtypes)

    #numerical_columns = dataset.describe().columns
    numerical_columns = dataset.select_dtypes(include=['number']).columns.tolist()

    return numerical_columns

def scale_numerical_columns(X_train, X_test):
    """scales numerical columns"""

    numerical_columns = get_numerical_columns(X_train)
    print(numerical_columns)

    scaler = preprocessing.StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    return X_train, X_test

def binary_encoding(X_train, X_test):
    """binary encodes relevent columns"""

    binary_columns = get_binary_columns(X_train) #potentially a problem if the sample contains only 2 values where it could be more in the whole dataset

    for column in binary_columns:
        encoder = preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        X_train[column] = encoder.fit_transform(X_train[[column]])
        X_test[column] = encoder.transform(X_test[[column]])

    return X_train, X_test

def one_hot_encoding(X_train, X_test):
    """one hot encodes relevent columns"""

    binary_columns = get_binary_columns(X_train)
    one_hot_columns = [column for column in categorielle if column not in binary_columns]

    encoder = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_array = encoder.fit_transform(X_train[one_hot_columns])

    X_train_encoded_df = pd.DataFrame(X_train_array, columns=encoder.get_feature_names_out(one_hot_columns), index=X_train.index)
    X_train = X_train.drop(columns=one_hot_columns).join(X_train_encoded_df)

    X_test_array = encoder.transform(X_test[one_hot_columns])

    X_test_encoded_df = pd.DataFrame(X_test_array, columns=encoder.get_feature_names_out(one_hot_columns), index=X_test.index)
    X_test = X_test.drop(columns=one_hot_columns).join(X_test_encoded_df)

    return X_train, X_test

def impute_dataset(X_train, X_test):
    """imputes missing values"""

    imputer = impute.SimpleImputer(strategy="most_frequent")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns= imputer.get_feature_names_out())
    X_test = pd.DataFrame(imputer.transform(X_test), columns= imputer.get_feature_names_out())

    return X_train, X_test

def get_coordinates_of_location(location):
    """calls nomi to get lat and lon on a specific location"""
    if re.match(r'^\d+$', str(location)[:5]): #postal code, we take the first 5 in case there are multiple : 75012, 75011
        result = nomi.query_postal_code(str(location)[:5])
        return result['latitude'], result['longitude']
    else: #region
        result = geolocator.geocode(str(location))
        return result.latitude, result.longitude

def get_dicos_coordinates(list_locations):
    """
    Based on a list returns a dico with unique values and their coordinates
    """
    dico = {}
    list_locations = list(set(list_locations))

    i = 0

    for elem in list_locations:
        i += 1
        if i%500 == 0: print(i, "/", len(list_locations))

        lat, lon = get_coordinates_of_location(elem)
        dico[elem] = [lat, lon]

    return dico

def preprocess_postal_code(df):
    """
    Add latitude and longitude columns to a DataFrame based on postal codes or regions.
    """
    # Create copies to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure lat and lon columns exist
    if 'lat' not in df.columns:
        df['lat'] = None
    if 'lon' not in df.columns:
        df['lon'] = None

    # Get dictionaries of coordinates
    print(datetime.now(), "get dico postal")
    dico_postal_code = get_dicos_coordinates(df["code_postal"])
    print(datetime.now(), "get dico region")
    dico_region = get_dicos_coordinates(df["region"])

    # First pass: Try to fill coordinates using postal codes
    mask_postal = df["code_postal"].isin(dico_postal_code.keys())

    # Use vectorized operations instead of iterating through rows
    postal_codes = df.loc[mask_postal, "code_postal"]

    print(datetime.now(), "fill code postal")
    # Update coordinates for postal codes
    for code in postal_codes.unique():
        mask = df["code_postal"] == code
        df.loc[mask, "lat"] = dico_postal_code[code][0]
        df.loc[mask, "lon"] = dico_postal_code[code][1]

    # Second pass: For rows that still have missing coordinates, try using region
    mask_missing = df["lat"].isna() | df["lon"].isna()

    print(datetime.now(), "fill region")
    if mask_missing.any():
        # Update coordinates for regions
        for region in dico_region:
            mask = (df["region"] == region) & mask_missing
            df.loc[mask, "lat"] = dico_region[region][0]
            df.loc[mask, "lon"] = dico_region[region][1]

    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    df.drop(columns=["code_postal", "region"], inplace=True)

    return df

def preprocess(dataset):

    #before train/test split
    print(datetime.now(), "preprocess forme juridique") #<------------- to be adapted so the logic is train/test and not whole dataset
    dataset = preprocessing_forme_juridique(dataset) #<------------- to be adapted so the logic is train/test and not whole dataset

    print(datetime.now(), "remove useless columns")
    dataset = remove_useless_columns(dataset)

    X, y = separate_x_y(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    #Missing values
    print(datetime.now(), "impute dataset")
    X_train, X_test = impute_dataset(X_train, X_test)

    #rebalancing of classes
    print(datetime.now(), "rebalance classes")
    print(datetime.now(), X_train.shape, y_train.shape)
    X_train, y_train = create_redistributed_dataset(X_train, y_train, 0.25)
    print(datetime.now(), X_train.shape, y_train.shape)

    print(datetime.now(), "preprocess nace")
    X_train = preprocessing_nace(X_train)
    X_test = preprocessing_nace(X_test)

    #postal code to lat/lon
    print(datetime.now(), "preprocess postal code")
    X_train = preprocess_postal_code(X_train)
    X_test = preprocess_postal_code(X_test)

    #v to remove later when we manage nace -------------------------
    X_train.drop(columns=["nace_four","nace_three","nace_two"], inplace=True) #will be removed later as we will use code postal and nace
    X_test.drop(columns=["nace_four","nace_three","nace_two"], inplace=True) #will be removed later as we will use code postal and nace

    print(datetime.now(), "format dates")
    X_train, X_test = format_date_creation(X_train, X_test)

    print(datetime.now(), "scale numerical columns")
    X_train, X_test = scale_numerical_columns(X_train, X_test)

    print(datetime.now(), "binary encoding")
    X_train, X_test = binary_encoding(X_train, X_test)

    print(datetime.now(), "one hot encoding")
    X_train, X_test = one_hot_encoding(X_train, X_test)

    return X_train, X_test, y_train, y_test

def main():

    X_train, X_test, y_train, y_test = preprocess(raw_dataset)

    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # print("----RECALL--")
    # results = cross_val_score(model, X_train, y_train, scoring=make_scorer(recall_score))
    # print(results.mean())
    # print("------")

    # _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    # # top 5 features that increase
    # top = 5
    # coeffs = list(model.coef_[0, :])
    # orders = sorted(coeffs, reverse=True)

    # for i in range(top):
    #     position = coeffs.index(orders[i])
    #     print("Name of the feature:", X_train.columns[position], "Coefficient:", orders[i])

    # # top 5 features that decrease
    # top = 5
    # coeffs = list(model.coef_[0, :])
    # orders = sorted(coeffs, reverse=False)

    # for i in range(top):
    #     position = coeffs.index(orders[i])
    #     print("Name of the feature:", X_train.columns[position], "Coefficient:", orders[i])

if __name__ == '__main__':
    main()


# def get_coordonnees_par_code(row):
#     lat, lon =  nomi.query_postal_code(str(row['code_postal']))['latitude'], nomi.query_postal_code(str(row['code_postal']))['longitude']
#     return  lat, lon
# def get_coordonnees_par_region(row):
#     lat, lon =  geolocator.geocode(str(row['region'])).latitude, geolocator.geocode(str(row['region'])).longitude
#     return  lat, lon
# # take unique code postal
# villes = pd.DataFrame(dataset.drop_duplicates(subset=['code_postal'])['code_postal'])
# villes['latitude'] = villes.apply(lambda row: get_coordonnees_par_code(row)[0], axis=1)
# villes['longitude'] = villes.apply(lambda row: get_coordonnees_par_code(row)[1], axis=1)
# region = pd.DataFrame(dataset.drop_duplicates(subset=['region'])['region'])
# region['latitude'] = region.apply(lambda row: get_coordonnees_par_region(row)[0], axis=1)
# region['longitude'] = region.apply(lambda row: get_coordonnees_par_region(row)[1], axis=1)
# # complete the dataset with the smaller coordonnes df
# dataset['latitude'] = dataset.apply(lambda row: region[region['region'] == row['region']]["latitude"].iloc[0] if (pd.isna(row['code_postal']) or pd.isna(villes[villes['code_postal'] == row['code_postal']]["latitude"].iloc[0])) else villes[villes['code_postal'] == row['code_postal']]["latitude"].iloc[0], axis=1)
# dataset['longitude'] = dataset.apply(lambda row: region[region['region'] == row['region']]["longitude"].iloc[0] if (pd.isna(row['code_postal']) or pd.isna(villes[villes['code_postal'] == row['code_postal']]["longitude"].iloc[0])) else villes[villes['code_postal'] == row['code_postal']]["longitude"].iloc[0], axis=1)
