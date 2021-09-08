import sys

import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from util import BERT

model_name = 'seBERT'
freeze_strategy = 'no_freeze'

checkpoint_dir = './checkpoints/ft_final'
batch_size = 8

clf = BERT(model_name, checkpoint_dir, freeze_strategy=freeze_strategy, batch_size=batch_size, num_labels=3)

# this time we load the full ground truth and only fit the model, then save it
df = pd.read_csv('../data/all_changes_gt.csv.gz')
X = df['message_no_newlines'].values
y = df['label'].values

clf.fit(X, y)
clf.save_model('./fine_tuned/')
