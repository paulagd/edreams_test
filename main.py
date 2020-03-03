# %matplotlib inline
from IPython import embed
import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import pandas as pd

import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import trange
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from dataset import eDreamsDataset
from nn_model import NeuralNetwork
from utils import write_csv, preprocess_data, evaluate, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DL", action="store_true", default=False, help="Enables deep learning training model.")
    parser.add_argument("--save_chkp", action="store_true", default=False, help="Enables weights to be saved.")
    parser.add_argument("--mean_encoding", action="store_true", default=False, help="Enables mean encoding (by default with DL).")
    parser.add_argument("--balanceDATA", action="store_false", default=True, help="Disables balancing classes.")
    parser.add_argument("--optimize", action="store_true", default=False, help="Enables optimization of ML algorithm.")
    # parser.add_argument("--learn_features", action="store_false", default=True, help="Enables learning features from emb")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs on training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate parameter.")
    parser.add_argument('--bs', type=int, default=256, help="Batch size parameter.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout factor.")
    parser.add_argument("--seed", type=int, default=1234, help="Fixed seed.")
    return parser.parse_args()


def DL_pipeline(x_train, y_train, x_val, y_val, args):
    '''
    Deep Learning pipeline of training-evaluation procedure
    '''

    train_set = eDreamsDataset(x_train, y_train)
    val_set = eDreamsDataset(x_val, y_val)
    
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=0)

    model_params = {'dropout': args.dropout,
                    'input_dim': X_train.shape[1],
                    }

    model = NeuralNetwork(**model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    global_step = 0
    early_stopping = 0
    last_val_loss = np.inf
    for j in trange(args.epochs, desc='Training NeuralNetwork...'):
        av_loss = []
        labels_train = []
        preds_train = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            
            # CALCULATE LOSS
            loss = criterion(y_pred, y.type(torch.FloatTensor))
            av_loss.append(loss.item())
            # BACKWARD PASS
            loss.backward()
            # MINIMIZE LOSS
            optimizer.step()
            labels_train.append(y)
            preds_train.append(y_pred.detach().numpy().squeeze())
            global_step += 1

        writer.add_scalar('train/loss', np.mean(av_loss), global_step)

        f1_train = f1_score(np.hstack(labels_train), np.round(np.hstack(preds_train)), average='weighted')
        writer.add_scalar('train/f1_score', f1_train, j)

        if j % 10 == 0:
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

            writer.add_scalar('val/loss', np.mean(val_loss), j)
            # EVALUATION
            f1 = f1_score(np.hstack(labels_arr), np.round(np.hstack(preds_arr)), average='weighted')
            
            print('[Training epoch {}/{}]: Loss = {}'.format(j, args.epochs, round(np.mean(av_loss), 3)))
            print('[Validation epoch {}] Loss: {} | f1 = {}'.format(j, round(np.mean(val_loss), 3), f1))

            writer.add_scalar('val/f1_score', f1, j)
            
            # IDEA: EARLY STOPPING
            if round(np.mean(val_loss), 3) >= round(last_val_loss, 3):
                if early_stopping > 5:
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


def ML_pipeline(X_train, y_train, X_val, model, param_grid, cv=10, scoring_fit='f1', do_probabilities=False):
    '''
    Machine Learning pipeline of training-evaluation procedure
    '''
    
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train, y_train)

    if do_probabilities:
        pred = fitted_model.predict_proba(X_val)
    else:
        pred = fitted_model.predict(X_val)

    return fitted_model, pred


if __name__ == '__main__':

    args = parse_args()
    date = datetime.now().strftime('%y%m%d%H%M%S')

    if args.DL:
        # If DL models --> Tensorboard for graphs visualization
        writer = SummaryWriter(log_dir=f'logs/logs_{date}/')

    # Checking if we have GPU
    if torch.cuda.is_available() and args.device:
        torch.cuda.manual_seed_all(args.seed)
        device = "cuda"
    else:
        device = "cpu"
        
    # Fixing seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(device)

    # Spliting in train, val and test. Also doing some preprocessing.
    main_path = '/Users/paulagomezduran/Desktop/EDREAMS/'
    X_train, y_train, X_val, y_val, X_test, df = preprocess_data(os.path.join(main_path, 'train.csv'),
                                                                 os.path.join(main_path, 'test.csv'), args,
                                                                 test_size=0.2, balanceDATA=args.balanceDATA)
    if args.DL:
        # Scalers for normalizing features (based on training data and applied to all data)
        scaler = MinMaxScaler(feature_range=(0, 1))  
        x_train = scaler.fit_transform(X_train.values)
        x_val = scaler.transform(X_val.values)
        x_test = scaler.transform(X_test.values)

        # Deep Learning pipeline --> returns trained model (evaluated with validation for optimization)
        model = DL_pipeline(x_train, y_train.values, x_val, y_val.values, args)

        # Making the final predictions on test data and writing the results in csv (ID, PREDICTION)
        model.eval()
        final_predictions = model(torch.Tensor(x_test)).detach().numpy().squeeze()
        write_csv(np.round(final_predictions), 'DL_finalSubmission')

    else:

        if args.optimize:
            model = lgb.LGBMClassifier()

            param_grid = {
                'n_estimators': [50, 100, 300, 500, 1000],
                'max_depth': [3, 10, 20, 30],
                'num_leaves': [10, 100, 1000, 2000],
                'learning_rate': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
            }

            model, y_pred = ML_pipeline(X_train, y_train, X_val, model, param_grid, cv=5, scoring_fit='f1')
            # roc_auc
            # It uses the best model parameters automatically
            print(model.best_score_)
            print(model.best_params_)
        else:
            model = lgb.LGBMClassifier(learning_rate=0.05, max_depth=30, n_estimators=1000, num_leaves=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        y_pred_prob = model.predict_proba(X_val)
        prob_y_1 = [p[1] for p in y_pred_prob]

        evaluate(y_pred, y_val.values, prob_y_1, True)

        final_predictions = model.predict(X_test)
        write_csv(np.round(final_predictions), 'LightGBM_finalSubmission')









