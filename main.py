# %matplotlib inline
from IPython import embed
import pandas as pd
import argparse
import torch

from datetime import datetime
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm, trange
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import eDreamsDataset
from nn_model import NeuralNetwork
from utils import mean_encoding


CATEGORICAL_FEATURES = ['WEBSITE', 'HAUL_TYPE', 'DEVICE', 'TRIP_TYPE', 'PRODUCT']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DL", action="store_true", default=False, help="Enables deep learning training model.")
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


def deep_learning_pipeline(X_train, y_train, X_val, y_val, args):
    # Generators
    train_set = eDreamsDataset(X_train.values, y_train.values)
    val_set = eDreamsDataset(X_val.values, y_val.values)
    
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=0)

    model_params = {'dropout': args.dropout,
                    'input_dim': X_train.shape[1],
                    }

    model = NeuralNetwork(**model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    global_step = 0
    for j in trange(args.epochs, desc='Training NeuralNetwork...'):

        for i, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)
            
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            
            # CALCULATE LOSS
            embed()
            # np.unique(y_pred.detach())
            loss = criterion(y_pred, y)
            # loss = criterion(pred, y.view(batch_size * seq_length).long())
            loss_value = loss.item()

            # BACKWARD PASS
            loss.backward()
            # MINIMIZE LOSS
            optimizer.step()
            global_step += 1
            writer.add_scalar('loss/train', loss_value, global_step)
            print('[Training epoch {}: {}/{}] Loss: {}'.format(j, i, len(train_loader), loss_value))

        if j % 10:
            #  EVALUATION EVERY 10 EPOCHS
            val_loss = []
            labels_arr = []
            preds_arr = []
            for (x, y) in enumerate(val_loader):

                model.eval()

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                # CALCULATE LOSS
                loss = criterion(y_pred, y)
                # loss = criterion(pred, y.view(batch_size * seq_length).long())
                # val_loss.append(loss.cpu().detach().numpy())
                labels_arr.append(y)
                preds_arr.append(y_pred)
                val_loss.append(loss.item())

            writer.add_scalar('loss/val', np.mean(val_loss), j)
            # EVALUATION
            f1, p, r = evaluation(labels_arr, preds_arr)
            
            print('[Validation epoch {}] Loss: {} | f1 = {}'.format(j, np.mean(val_loss), f1))

            writer.add_text('evaluation/f1_score', f1, j)
            writer.add_text('evaluation/precision', p, j)
            writer.add_text('evaluation/recall', r, j)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        os.makedirs("weights/{}".format(date), exist_ok=True)
        torch.save(checkpoint, "weights/{}/checkpoint_{}.pt".format(date, j))
        

if __name__ == '__main__':

    args = parse_args()

    # for visualization
    date = datetime.now().strftime('%y%m%d%H%M%S')
    if args.DL:
        writer = SummaryWriter(log_dir=f'logs/logs_{date}/')
    else:
        writer = SummaryWriter(log_dir=f'logs/nologs/logs/')

    if torch.cuda.is_available() and args.device:
        torch.cuda.manual_seed_all(args.seed)
        device = "cuda"
    else:
        device = "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(device)

    X_train, y_train, X_val, y_val, X = preprocess_data('/Users/paulagomezduran/Desktop/EDREAMS/train.csv', True, args)
    # df_test = filter_data('/Users/paulagomezduran/Desktop/EDREAMS/test.csv', False)

    if args.DL:
        deep_learning_pipeline(X_train, y_train, X_val, y_val, args)


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






