

def get_columns_with_nan(df):
    return df.columns[df.isna().any()].tolist()

def get_numeric_columns(df):
    return df.dtypes[df.dtypes != "object"].index

def get_categorical_columns(df):
    return df.dtypes[df.dtypes == "object"].index






