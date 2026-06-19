import os
import pandas as pd

def fill_not_applicable(df: pd.DataFrame):
    """
    Replace structural missing values (-9) with a valid "Not applicable" category (0).

    Some variables are undefined by design for certain patients (e.g., pregnancy for males).
    In such cases, -9 indicates "Not applicable" rather than missing information.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with structural missing values corrected.
    """
    df.loc[(df['DETCRIM'] == -9) & (df['PSOURCE'] != 7), 'DETCRIM'] = 0
    df.loc[(df['DETNLF'] == -9) & (df['EMPLOY'] != 4), 'DETNLF'] = 0
    df.loc[(df['DETNLF_D'] == -9) & (df['EMPLOY_D'] != 4), 'DETNLF_D'] = 0
    df.loc[(df['PREG'] == -9) & (df['GENDER'] != 2), 'PREG'] = 0
    return df

def _fill_help(df:pd.DataFrame, sub, target_var):
    """
    Helper function to fill dependency-based missing values (-9).

    If an upstream indicator (`sub`) is active and the downstream variable
    is missing, the downstream value is set to 0.

    Args:
        df (pd.DataFrame): Input DataFrame.
        sub (str): Upstream indicator column.
        target_var (str): Downstream column to fill.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    df.loc[(df[sub] == 1) & (df[target_var] == -9), target_var] = 0
    return df

def fill_not_available(df:pd.DataFrame):
    """
    Replace dependency-driven missing values (-9) with a valid "Unavailable" category (0).

    Some variables are missing due to procedural dependencies in data collection.
    This function resolves such cases for substance-related variables at both
    admission and discharge.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with dependency-based missing values corrected.
    """
    variables = ('SUB', 'FREQ', 'ROUTE', 'FRSTUSE')
    for i in ('1', '2', '3'):
        cur_var = [name + i for name in variables]

        # admission variable
        df = _fill_help(df, cur_var[0], cur_var[1])
        df = _fill_help(df, cur_var[0], cur_var[2])
        df = _fill_help(df, cur_var[0], cur_var[3])

        # discharge variable '_D'
        df = _fill_help(df, cur_var[0] + '_D', cur_var[1] + '_D')
    return df

def tackle_missing_value(raw_df_path: str):
    """
    Load raw data and correct both structural and dependency-based missing values.

    Args:
        raw_df_path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: DataFrame with corrected missing values.
    """
    raw_df = pd.read_csv(raw_df_path)
    missing_corrected = fill_not_applicable(raw_df)
    missing_corrected = fill_not_available(missing_corrected)
    return missing_corrected

def tackle_missing_value_wrapper(raw_df_path: str, missing_corrected_path: str):
    """
    Load preprocessed data if available; otherwise preprocess and save it.

    Args:
        raw_df_path (str): Path to the raw CSV file.
        missing_corrected_path (str): Path to save/load the processed CSV.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    if os.path.exists(missing_corrected_path):
        missing_corrected = pd.read_csv(missing_corrected_path)
        return missing_corrected
    
    df = tackle_missing_value(raw_df_path)
    
    os.makedirs(os.path.dirname(missing_corrected_path) or ".", exist_ok=True)
    df.to_csv(missing_corrected_path, index=False)
    return df