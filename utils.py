import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from IPython import embed


def mean_encoding(df, col):
    # column_dict = {x: i for i, x in enumerate(df[col].unique())}
    column_dict = df.groupby(col).mean()['EXTRA_BAGGAGE'].to_dict()
    df[col] = df[col].apply(column_dict.get)
    return df


def feature_selection(df, X, y):
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

