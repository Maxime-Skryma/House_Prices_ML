import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from scipy.stats import f_oneway
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def preprocessing(df,power_transformer=None,imputer=None):
    df = df.drop(columns=['Id'])
    
    #On fill les NaN
    cat_cols = df.select_dtypes(exclude='number').columns
    df[cat_cols] = df[cat_cols].fillna('None')

    #On crée une catégories Pool (apporte +d'infos)
    df['Pool']=(df['PoolArea']>0).astype(float)
    df = df.drop(columns=['PoolArea'])

    #Modification des variables catégorielles qui sont en fait ordinales

    # Échelle de qualité générique (revient sur 6 colonnes)
    qual_scale = ['Po', 'Fa', 'TA', 'Gd', 'Ex']

    # Variantes avec "pas d'objet" (NA) en plus, à placer selon ton choix
    # (toi tu as déjà remplacé les NA par 'None' plus tôt dans la conversation)
    qual_scale_with_none = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

    # PoolQC n'a pas de modalité 'Po' dans le fichier
    poolqc_scale = ['None', 'Fa', 'TA', 'Gd', 'Ex']

    ordinal_scales = {
        'ExterQual':    qual_scale_with_none,   # pas de NA en réalité, mais safe
        'ExterCond':    qual_scale_with_none,
        'HeatingQC':    qual_scale_with_none,
        'KitchenQual':  qual_scale_with_none,
        'BsmtQual':     qual_scale_with_none,   # NA = No Basement
        'GarageQual':   qual_scale_with_none,   # NA = No Garage
        'GarageCond':   qual_scale_with_none,   # NA = No Garage
        'FireplaceQu':  qual_scale_with_none,   # NA = No Fireplace
        'PoolQC':       poolqc_scale,           # NA = No Pool, pas de 'Po'

        'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],           # NA = No Basement
        'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],  # NA = No Basement
        'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],

        'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],              # NA = No Garage

        'Utilities':   ['None', 'ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
        'LotShape':    ['None', 'Reg', 'IR1', 'IR2', 'IR3'],
        'LandSlope':   ['None', 'Gtl', 'Mod', 'Sev'],
        'PavedDrive':  ['None', 'N', 'P', 'Y'],
        'Functional':  ['None', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    

        'Fence':        ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],  # ordre discutable, cf. remarque
    }
    
    #On 'ordinalise' et on 'dummies' ce qui reste

    for c in ordinal_scales:
        mapping = {cat: i for i, cat in enumerate(ordinal_scales[c])}
        df[c] = df[c].map(mapping)
        if df[c].isna().any():
            raise ValueError(f"{c} : modalités non mappées → {df.loc[df[c].isna(), c]}")


    reste_col=df.select_dtypes(exclude='number').columns
    df = pd.get_dummies(df, columns=reste_col, drop_first=True)

    if imputer is None:
        train_columns = df.columns
    else:
        df = df.reindex(columns=imputer.train_columns, fill_value=0)

    #On transforme les variables avec un skew trop élevé avec la méthode yeo-johnson
    cat_cols = df.select_dtypes(include='number').columns

    if power_transformer is None:
        skew_cols = [c for c in cat_cols if (abs(df[c].skew()) > 1 and abs(df[c].skew()) < 5) ]
        power_transformer = PowerTransformer(method='yeo-johnson',standardize=False)
        df[skew_cols] = power_transformer.fit_transform(df[skew_cols])
    else:
        skew_cols = list(power_transformer.feature_names_in_)
        df[skew_cols] = power_transformer.transform(df[skew_cols])
    
    df_imputer=df.select_dtypes(include=["number","bool"])

    if imputer is None:
        imputer = IterativeImputer(random_state=42)
        df = pd.DataFrame(imputer.fit_transform(df_imputer),
                        columns=df_imputer.columns, index=df_imputer.index)
        imputer.train_columns = train_columns
    else:
        df = pd.DataFrame(imputer.transform(df_imputer),
                        columns=df_imputer.columns, index=df_imputer.index)


    return df, power_transformer , imputer
