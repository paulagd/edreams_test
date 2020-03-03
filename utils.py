import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
from datetime import datetime
import csv
import pandas as pd
from IPython import embed
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


def create_tripDuration(df, df_set):

    for i, d in tqdm(enumerate(df['ARRIVAL']), desc=f"Converting ARRIVAL str to datetime [{df_set}]",
                     total=len(df['ARRIVAL'])):
        df['ARRIVAL'][i] = datetime.strptime(d, '%d/%B')

    for i, d in tqdm(enumerate(df['DEPARTURE']), desc=f"Converting DEPARTURE str to datetime [{df_set}]",
                     total=len(df['DEPARTURE'])):
        df['DEPARTURE'][i] = datetime.strptime(d, '%d/%B')

    df['TRIP_DURATION'] = df['ARRIVAL'].sub(df['DEPARTURE'], axis=0).dt.days
    neg_values = df[df['TRIP_DURATION'] < 0]['TRIP_DURATION'] + 365
    df.update(neg_values)
    return df.drop(columns=['DEPARTURE', 'ARRIVAL', 'TIMESTAMP'])


def mean_encoding(df, col, column_dict=None):
    if column_dict is None:
        column_dict = df.groupby(col).mean()['EXTRA_BAGGAGE'].to_dict()
    df[col] = df[col].apply(column_dict.get)
    return column_dict, df


def write_csv(preds, title):
    header = ['ID', 'EXTRA_BAGGAGE']

    df = pd.DataFrame(np.stack((range(30000), preds.astype(np.bool)), axis=-1), columns=header)
    df = df['EXTRA_BAGGAGE'].astype(np.bool)
    df.to_csv(f'{title}_pandas.csv')


def preprocess_data(dataset_train, dataset_test, args, test_size=0.2, balanceDATA=True):

    me_flag = True if (args.mean_encoding or args.DL) else False
    
    df = pd.read_csv(dataset_train, sep=';', engine='python')
    df_test = pd.read_csv(dataset_test, sep=';', engine='python')

    ##########################################################################################
    # IDEA: I substitute ARRIVAL AND DEPARTURE DATES BY TRIP_DURATION, WHICH I THINK IS MORE RELEVANT
    df = create_tripDuration(df, 'training set')
    df_test = create_tripDuration(df_test, 'test set')
    ##########################################################################################
    # IDEA: I convert the distance in integer as well
    df['DISTANCE'] = df['DISTANCE'].str.replace(',', '').astype(int)
    df_test['DISTANCE'] = df_test['DISTANCE'].str.replace(',', '').astype(int)
    # IDEA: I fill nan values of DEVICE feature as 'OTHER' options
    df = df.fillna(value={'DEVICE': 'OTHER'})
    df_test = df_test.fillna(value={'DEVICE': 'OTHER'})
    ##########################################################################################
    if balanceDATA:
        df_majority = df[df.EXTRA_BAGGAGE == 0]
        df_minority = df[df.EXTRA_BAGGAGE == 1]
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=1234)
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        df = df_upsampled
    #########################################################################################
    # IDEA: set categorical variables in pandas for ML model
    if not me_flag:
        for col in ['WEBSITE', 'HAUL_TYPE', 'DEVICE', 'TRIP_TYPE', 'PRODUCT']:
            df[col] = df[col].astype('category')
            df_test[col] = df_test[col].astype('category')

    # DIVIDE DATA INTO TRAIN / VAL  --> DATA/LABELS
    train_df, val_df = train_test_split(df, test_size=test_size)
    if me_flag:
        # MEAN ENCODING FOR CATEGORICAL VARIABLES
        # With train encoding parameters --> apply to validation and test set
        for col in ['WEBSITE', 'HAUL_TYPE', 'DEVICE', 'TRIP_TYPE', 'PRODUCT']:
            column_dict, train_df = mean_encoding(train_df, col)
            # SUBSTITUTE IN VAL SET
            _, val_df = mean_encoding(val_df, col, column_dict)
            val_df[col].fillna((train_df[col].mean()), inplace=True)
            # SUBSTITUTE IN TEST SET
            _, df_test = mean_encoding(df_test, col, column_dict)
            df_test[col].fillna((train_df[col].mean()), inplace=True)

        train_df = train_df.astype(float)
        assert val_df.isnull().values.any() == False, 'There are NaN values in validation set'
        val_df = val_df.astype(float)
        df_test = df_test.astype(float)

    y_train = train_df['EXTRA_BAGGAGE']
    X_train = train_df.drop('EXTRA_BAGGAGE', 1)
    y_val = val_df['EXTRA_BAGGAGE']
    X_val = val_df.drop('EXTRA_BAGGAGE', 1)

    return X_train, y_train, X_val, y_val, df_test, df


def evaluate(y_pred, y_true, prob_y_1):

    n_errors = len(np.where((y_pred == y_true) == False)[0])
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, prob_y_1)

    print('Model Performance')
    print('-----------------------------')
    print('Error percentage: {:0.1f} %'.format((n_errors / len(y_pred)) * 100))
    print('F1 score = {:0.2f}%'.format(f1 * 100))
    print('ROC_AUC = {:0.2f}%'.format(roc * 100))
    print('-----------------------------')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, normalize=True)


def plot_confusion_matrix(y_true, y_pred, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        title = 'Normalized confusion matrix'
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        title = 'Confusion matrix, without normalization'
        print('Confusion matrix, without normalization')

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['SINGLE BAG', 'EXTRA BAG'])
    ax.yaxis.set_ticklabels(['SINGLE BAG', 'EXTRA BAG'])
    plt.show()