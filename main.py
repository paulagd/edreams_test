# %matplotlib inline
from IPython import embed
import pandas as pd
import argparse
import torch

from datetime import datetime
import numpy as np
import os
import torch.nn as nn
from tqdm import trange
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import eDreamsDataset
from nn_model import NeuralNetwork
from utils import mean_encoding, create_tripDuration


CATEGORICAL_FEATURES = ['WEBSITE', 'HAUL_TYPE', 'DEVICE', 'TRIP_TYPE', 'PRODUCT']


def preprocess_data(dataset_train, dataset_test, args):
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
    # IDEA: set categorical variables in pandas for ML model
    if not args.DL:
        for col in ['WEBSITE', 'HAUL_TYPE', 'DEVICE', 'TRIP_TYPE', 'PRODUCT']:
            df[col] = df[col].astype('category')
            df_test[col] = df_test[col].astype('category')

    # DIVIDE DATA INTO TRAIN / VAL  --> DATA/LABELS
    train_df, val_df = train_test_split(df, test_size=0.2)
    if args.DL:
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

    return X_train, y_train, X_val, y_val, df_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DL", action="store_true", default=False, help="Enables deep learning training model.")
    parser.add_argument("--save_chkp", action="store_true", default=False, help="Enables deep learning training model.")
    # parser.add_argument("--learn_features", action="store_false", default=True, help="Enables learning features from emb")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs on training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate parameter.")
    parser.add_argument('--bs', type=int, default=256, help="Batch size parameter.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout factor.")
    parser.add_argument("--seed", type=int, default=1234, help="Fixed seed.")
    return parser.parse_args()


def evaluation(labels, preds):
    f1 = f1_score(labels, preds, average="samples")
    p = precision_score(labels, preds, average="samples")
    r = recall_score(labels, preds, average="samples")

    return f1, p, r


def DL_pipeline(X_train, y_train, X_val, y_val, X_test, args):
    '''
    Deep Learning pipeline of training-evaluation procedure
    '''
    # Generators
    scaler = MinMaxScaler()

    train_set = eDreamsDataset(scaler.fit_transform(X_train.values), y_train.values)
    val_set = eDreamsDataset(scaler.fit_transform(X_val.values), y_val.values)
    
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=0)

    model_params = {'dropout': args.dropout,
                    'input_dim': X_train.shape[1],
                    }

    model = NeuralNetwork(**model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    global_step = 0
    early_stopping = 0
    last_val_loss = np.inf
    for j in trange(args.epochs, desc='Training NeuralNetwork...'):
        av_loss = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            
            # CALCULATE LOSS
            # np.unique(y_pred.detach())
            loss = criterion(y_pred, y.type(torch.FloatTensor))
            av_loss.append(loss.item())
            # BACKWARD PASS
            loss.backward()
            # MINIMIZE LOSS
            optimizer.step()
            global_step += 1

        writer.add_scalar('loss/train', np.mean(av_loss), global_step)
        print('[Training epoch {}/{}]: Loss = {}'.format(j, args.epochs, np.mean(av_loss)))

        if j % 10:
            #  EVALUATION EVERY 10 EPOCHS
            val_loss = []
            labels_arr = []
            preds_arr = []
            for i_val, (x, y) in enumerate(val_loader):
                model.eval()
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                # CALCULATE LOSS
                loss = criterion(y_pred, y.type(torch.FloatTensor))
                # APPEND LABELS AND PREDICTIONS FOR EVALUATIONS
                labels_arr.append(y)
                preds_arr.append(y_pred.detach().numpy().squeeze())
                val_loss.append(loss.item())

            writer.add_scalar('loss/val', np.mean(val_loss), j)
            # EVALUATION
            f1 = f1_score(np.hstack(labels_arr), np.round(np.hstack(preds_arr)), average="binary")
            # f1, p, r = evaluation(labels_arr, preds_arr)
            
            # print('[Validation epoch {}]: Loss = {} | '.format(j, np.mean(val_loss)))
            print('[Validation epoch {}] Loss: {} | f1 = {}'.format(j, np.mean(val_loss), f1))

            writer.add_scalar('evaluation/f1_score', f1, j)
            # writer.add_scalar('evaluation/precision', p, j)
            # writer.add_scalar('evaluation/recall', r, j)
            
            # IDEA: EARLY STOPPING
            if np.mean(val_loss) > last_val_loss:
                if early_stopping > 30:
                    break
                early_stopping += 1
            else:
                early_stopping = 0

            last_val_loss = np.mean(val_loss)
            
            if args.save_chkp:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                os.makedirs("weights/{}".format(date), exist_ok=True)
                torch.save(checkpoint, "weights/{}/checkpoint_{}.pt".format(date, j))

    return model
        

def ML_pipeline(X_train, y_train, X_val, y_val, X_test, args):
    '''
    Machine Learning pipeline of training-evaluation procedure
    '''


if __name__ == '__main__':

    args = parse_args()

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    if args.DL:
        writer = SummaryWriter(log_dir=f'logs/logs_{date}/')

    if torch.cuda.is_available() and args.device:
        torch.cuda.manual_seed_all(args.seed)
        device = "cuda"
    else:
        device = "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(device)

    main_path = '/Users/paulagomezduran/Desktop/EDREAMS/'

    X_train, y_train, X_val, y_val, X_test = preprocess_data(os.path.join(main_path, 'train.csv'),
                                                             os.path.join(main_path, 'test.csv'), args)

    if args.DL:
        model = DL_pipeline(X_train, y_train, X_val, y_val, X_test, args)
        embed()
    else:
        ML_pipeline(X_train, y_train, X_val, y_val, X_test, args)


    
    #######################################################################################################
    # CREATE A MODEL
    # embed()
    # lgb_train = lgb.Dataset(X_train, y_train)
    # lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    #
    # # specify your configurations as a dict
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2', 'l1'},
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 0
    # }

    # # gbm0 = lgb.train(params, lgb_train, num_boost_round=10, verbose_eval=False)
    # print('Starting training...')
    # # train
    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=20,
    #                 valid_sets=lgb_eval,
    #                 early_stopping_rounds=5)
    #
    # print('Saving model...')
    # # save model to file
    # gbm.save_model('model.txt')
    #
    # print('Starting predicting...')
    # # predict
    # y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    # # eval
    # print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)
    ############################################################################################
    # fit_params = {"early_stopping_rounds": 10,
    #               "eval_metric": 'auc',
    #               "eval_set": [(X_val, y_val)],
    #               'eval_names': ['valid'],
    #               'verbose': 100,
    #               'feature_name': 'auto',  # that's actually the default
    #               'categorical_feature': 'auto'  # that's actually the default
    #               }
    # clf = lgb.LGBMClassifier(num_leaves=16, max_depth=-1,
    #                          random_state=314,
    #                          silent=True,
    #                          metric='None',
    #                          n_jobs=4,
    #                          n_estimators=1000,
    #                          colsample_bytree=0.9,
    #                          subsample=0.9,
    #                          learning_rate=0.1)
    #
    # clf.fit(X_train, y_train, **fit_params)
    # clf.predict
    # feat_imp = pd.Series(clf.feature_importances_, index=X.columns[1:])
    # # embed()
    # feat_imp.nlargest(16).plot(kind='barh', figsize=(8,10))
    # plt.show()
    # embed()
    # [X_train, y_train].corr()






