import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# load ground truth data
df = pd.read_csv('../data/all_changes_gt.csv.gz')
X = df['message_no_newlines'].values
y = df['label'].values

# 10x10 cross-validation
for run_number in range(10):
    np.random.seed(run_number)
    sp = KFold(n_splits=10, shuffle=True)
    for fold_number, (train_idx, test_idx) in enumerate(sp.split(X, y)):

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        with open('folds/{}_{}_X_train_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, X_train)
        with open('folds/{}_{}_X_test_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, X_test)
        with open('folds/{}_{}_y_train_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, y_train)
        with open('folds/{}_{}_y_test_commit_intent.npy'.format(run_number, fold_number), 'wb') as f:
            np.save(f, y_test)
