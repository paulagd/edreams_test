import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from tqdm import tqdm
from datetime import datetime
from IPython import embed


def create_tripDuration(df, df_set):
    # df['ARRIVAL'] = df['ARRIVAL'].apply(lambda x: datetime.strptime(x, '%d/%B') * 2)

    # for index, row in tqdm(df.iterrows()):
    #     setattr(row, "ARRIVAL", datetime.strptime(getattr(row, "ARRIVAL"), '%d/%B'))
        # df.at['ARRIVAL', index] = datetime.strptime(getattr(row, "ARRIVAL"), '%d/%B')

    for i, d in tqdm(enumerate(df['ARRIVAL']), desc=f"Converting ARRIVAL str to datetime [{df_set}]",
                     total=len(df['ARRIVAL'])):
        df['ARRIVAL'][i] = datetime.strptime(d, '%d/%B')
        # df.loc['ARRIVAL', i] = datetime.strptime(d, '%d/%B')
        # df.at['ARRIVAL', i] = datetime.strptime(d, '%d/%B')

    for i, d in tqdm(enumerate(df['DEPARTURE']), desc=f"Converting DEPARTURE str to datetime [{df_set}]",
                     total=len(df['DEPARTURE'])):
        df['DEPARTURE'][i] = datetime.strptime(d, '%d/%B')
        # df.loc['DEPARTURE', i] = datetime.strptime(d, '%d/%B')

    df['TRIP_DURATION'] = df['ARRIVAL'].sub(df['DEPARTURE'], axis=0).dt.days
    neg_values = df[df['TRIP_DURATION'] < 0]['TRIP_DURATION'] + 365
    df.update(neg_values)
    return df.drop(columns=['DEPARTURE', 'ARRIVAL', 'TIMESTAMP'])


def mean_encoding(df, col, column_dict=None):
    # column_dict = {x: i for i, x in enumerate(df[col].unique())}
    if column_dict is None:
        column_dict = df.groupby(col).mean()['EXTRA_BAGGAGE'].to_dict()
    df[col] = df[col].apply(column_dict.get)
    # df[col] = df[col].astype('category')
    return column_dict, df


def feature_selection(df, X, y):
    # df['EXTRA_BAGGAGE'].value_counts()

    FEATURES_MAP = np.asarray(['ID', 'TIMESTAMP', 'WEBSITE', 'GDS', 'ADULTS', 'CHILDREN', 'INFANTS', 'TRAIN',
                               'HAUL_TYPE', 'DISTANCE', 'DEVICE', 'TRIP_TYPE', 'PRODUCT', 'SMS', 'NO_GDS',
                               'TRIP_DURATION'])
    ##########################################################################################
    # IDEA: Now that all the data are integer values we can calculate the Pearson Corralation
    # Using Pearson Correlation
    plt.figure(figsize=(12, 10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    # Correlation with output variable
    cor_target = abs(cor["EXTRA_BAGGAGE"])
    ax = abs(cor["EXTRA_BAGGAGE"]).plot.bar()
    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.1]
    #
    # # IDEA: I split the data from their labels
    # Y = df['EXTRA_BAGGAGE']
    # X = df.drop('EXTRA_BAGGAGE', 1)
    #
    # embed()
    # df[['TRIP_DURATION', 'EXTRA_BAGGAGE']].groupby(['TRIP_DURATION'], as_index=False).mean().sort_values(
    #     by='EXTRA_BAGGAGE', ascending=False)

    # Feature extraction
    test = SelectKBest(score_func=chi2, k=5)
    fit = test.fit(X, y)

    # Summarize scores
    np.set_printoptions(precision=3)
    # print(fit.scores_)

    idx_selected_features = fit.scores_.argsort()[-5:][::-1]
    print(f'SELECTED FEATURES ARE: {FEATURES_MAP[np.sort(idx_selected_features)]}')

    features = fit.transform(X)
    return features

