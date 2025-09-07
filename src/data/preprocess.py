import json
import re
import pyarrow
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.tree_utils import find_invalid_expressions

def load_interim_data(folder_name):
    """
    Loads all .json files from a given subfolder under data/raw/.

    Args:
        folder_name (str): The name of the subfolder inside data/raw (e.g. 'elementary', 'special').

    Returns:
        list: A list of JSON-parsed objects from each file.
    """
    root = Path(__file__).resolve().parents[2]  # "Journal Paper Final/"
    target_folder = root / "data" / "interim" / folder_name

    if not target_folder.exists():
        raise FileNotFoundError(f"The folder {target_folder} does not exist.")
    
    json_data_list = []
    for file_path in target_folder.glob("*.json"):
        print(f"Loading file: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
            json_data_list.append(data)
    
    return json_data_list


def replace_int_with_C(L):
    """
    Replaces integer string elements in a list with constant tokens, except for integers in the range [-2, 2].
    Args:
        L (list of str): List of strings representing a prefix notation expression.
    Returns:
        list of str: A new list where integer strings outside [-2, 2] are replaced CONST tokens
    """   
    keep_list = list(range(-2,3)) # dont replace these integers in[-2, 2] with CONST
    new_L = L.copy()
        
    for i in range(len(L)):
        if L[i].isdigit():
            
            if int(L[i]) not in keep_list:
                if len(L[i])==1: # 1 digit integers
                    new_L[i] = 'CONST1'
                elif len(L[i])==2: # 2 digit integers
                    new_L[i] = 'CONST2'
                else: # all other cases
                    new_L[i] = 'CONST3'
    return new_L


# Model does not handle the following cases correctly, so we need to filter them out
def contains_unsupported_expressions(s):
    # Pattern for abs(x, y) or Complex(x, y) with two arguments separated by a comma
    abs_complex_pattern = r"(abs|Complex)\s*\([^)]*,[^)]*\)"
    # Pattern for integer complex numbers of the form n + m*I, including cases where m=1 (e.g., "2 + I")
    complex_number_pattern = r"\b-?\d+\s*\+\s*-?(?:\d+\*)?I\b"

    # Return True if either pattern is found
    return bool(re.search(abs_complex_pattern, s)) or bool(re.search(complex_number_pattern, s))


def min_max_scale(L):
    """
    Scale L so that its non-(-1) values map from [min, max] → [0,1].
    Any element equal to -1 is left unchanged.
    If all valid values are identical (zero range), they become 0.
    """
    # Collect only the “valid” values
    valid = [x for x in L if x != -1]
    # If no valid entries, nothing to scale
    if not valid:
        return L.copy()

    min_val = min(valid)
    max_val = max(valid)
    range_val = max_val - min_val

    out = []
    for x in L:
        if x == -1:
            # leave missing label alone
            out.append(-1)
        else:
            if range_val == 0:
                # zero range → collapse to 0
                out.append(0.0)
            else:
                out.append((x - min_val) / range_val)
    return out


def build_vocab(series):
    """
    Builds a vocabulary dictionary from a series of token lists.
    The vocabulary will include two special tokens:
        '<pad>': 0  - Used for padding sequences.
        '<OOV>': 1  - Used for out-of-vocabulary tokens.
    Each unique token in the input series is assigned a unique integer index starting from 2.
    Args:
        series (Iterable[Iterable[str]]): An iterable of token lists (e.g., list of sentences, each as a list of tokens).
    Returns:
        dict: A dictionary mapping tokens to unique integer indices.
    """
    vocab = {'<OOV>': 1,
             '<pad>': 0}  # Out of vocabulary
    index = 2
    
    for tokens in series:
        for token in tokens:
            if token not in vocab:  # If token is not already in the vocabulary
                vocab[token] = index  # Assign a new index
                index += 1
    return vocab

# TODO: combine elementary and special data into one return, new column for 'data_type' to distinguish them
# TODO: create vocab.json. it already exists, but it should be done here

data_list_train = load_interim_data("train/elementary")
data_list_train_nonelem = load_interim_data("train/nonelementary")
data_list_test = load_interim_data("test/elementary")
data_list_test_nonelem = load_interim_data("test/nonelementary")

train_data = [example for sublist in data_list_train for example in sublist] # combine all 6 data sources into one list
train_data_nonelem = [example for sublist in data_list_train_nonelem for example in sublist] # combine all 6 data sources into one list
test_data = [example for sublist in data_list_test for example in sublist]
test_data_nonelem = [example for sublist in data_list_test_nonelem for example in sublist]

# HACK: data from fwd, bwd, ibp have 5 items (integral in prefix). get rid of that to match len of other data
train_data = [sublist[:3] + sublist[4:] if len(sublist) == 5 else sublist for sublist in train_data]
test_data = [sublist[:3] + sublist[4:] if len(sublist) == 5 else sublist for sublist in test_data]

# Merge elem and nonelem lists

df_train = pd.DataFrame(train_data, columns=['integrand', 'prefix', 'integral', 'label_original'])
df_test = pd.DataFrame(test_data, columns=['integrand', 'prefix', 'integral', 'label_original'])
df_train['source'] = 'elementary'
df_test['source'] = 'elementary'
print("train length:", len(df_train))
df_nonelem_train = pd.DataFrame(train_data_nonelem, columns=['integrand', 'prefix', 'integral', 'label_original'])
df_nonelem_test = pd.DataFrame(test_data_nonelem, columns=['integrand', 'prefix', 'integral', 'label_original'])
df_nonelem_train['source'] = 'nonelementary'
df_nonelem_test['source'] = 'nonelementary'
df_train = pd.concat([df_train, df_nonelem_train], ignore_index=True)
df_test = pd.concat([df_test, df_nonelem_test], ignore_index=True)
print("train length after merge:", len(df_train))

# Remove label lookup
df_train['label_original'] = df_train['label_original'].apply(lambda arr: np.delete(arr, 10))
df_test['label_original'] = df_test['label_original'].apply(lambda arr: np.delete(arr, 10))

# Glitch when labelled data produces an error and removes its labels: delete from dataset
df_train = df_train[df_train['label_original'].apply(lambda x: len(x) == 12)]
print("Removed lookup")

# replace all integers in prefix with a CONST. Remove duplicates after this substitution
df_train['prefix'] = df_train['prefix'].apply(replace_int_with_C)
df_train["prefix"] = df_train["prefix"].transform(lambda k: tuple(k)) # transforming to tuple is much faster operation
df_train.drop_duplicates(subset='prefix', inplace=True)
df_train['prefix'] = df_train['prefix'].apply(list)

df_test['prefix'] = df_test['prefix'].apply(replace_int_with_C)

# Remove unspported expressions
df_train = df_train[~ df_train['integrand'].apply(contains_unsupported_expressions)]
df_test = df_test[~ df_test['integrand'].apply(contains_unsupported_expressions)]

# Get a list of invalid expressions, remove them since we cant deal with them
errors_train = find_invalid_expressions(df_train['prefix'])
errors_test = find_invalid_expressions(df_test['prefix'])

df_train = df_train[~ df_train['prefix'].isin(errors_train)]
df_test = df_test[~ df_test['prefix'].isin(errors_test)]

# Add a CLS token to the beginning of each prefix (from BERT paper)
df_train['prefix'] = df_train['prefix'].apply(lambda x: ['[CLS]'] + x)
df_test['prefix'] = df_test['prefix'].apply(lambda x: ['[CLS]'] + x)

# scale labels to [0,1] range
df_train['label'] = df_train['label_original'].apply(min_max_scale)
df_test['label'] = df_test['label_original'].apply(min_max_scale)

# save to parquet
# Make sure to use __file__ to get the location of the script, not the working directory!
processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

table_train = pyarrow.Table.from_pandas(df_train)
table_test = pyarrow.Table.from_pandas(df_test)

pq.write_table(table_train, processed_dir / "train_data_new.parquet")
pq.write_table(table_test, processed_dir / "test_data_new.parquet")
print("processed data")
  