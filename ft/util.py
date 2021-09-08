import os
import gc
import threading
import sys

import numpy as np
import torch

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

def compute_metrics_multi_label(p):
    """This metrics computation is only for finetuning."""
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro')
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)
    
    recall_ma = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision_ma = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1_ma = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {'accuracy': accuracy, 'precision_micro': precision, 'recall_micro': recall, 'f1_micro': f1, 'mcc': mcc, 'precision_macro': precision_ma, 'recall_macro': recall_ma, 'f1_macro': f1_ma}


def load_fold_multi_label(run_number, fold_number):
    """Load pre-generated data for run and fold numbers in case of multi-label classification."""
    with open('folds/{}_{}_X_train_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
    with open('folds/{}_{}_X_test_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        X_test = np.load(f, allow_pickle=True)
    with open('folds/{}_{}_y_train_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)
    with open('folds/{}_{}_y_test_commit_intent.npy'.format(run_number, fold_number), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)
    return X_train, y_train, X_test, y_test


def get_model_and_tokenizer(model_name, num_labels=2):
    """Load model_name and tokenizer, assumes paths from the Readme"""
    model = None
    tokenizer = None
    if model_name == 'seBERT':
        model = BertForSequenceClassification.from_pretrained('./models/seBERT/pytorch_model.bin', config='./models/seBERT/config.json', num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('./models/seBERT/', do_lower_case=True)
    else:
        raise Exception('model {} not implemented'.format(model_name))
    return model, tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class BERT(BaseEstimator, ClassifierMixin):
    """This wraps different BERT models with the help of the huggingface transformers library into a classifier for scikit-learn."""
    def __init__(self, model_name, checkpoints_dir, freeze_strategy=None, batch_size=8, num_labels=2):
        self.model_name = model_name
        self.trainer = None
        self.checkpoints_dir = checkpoints_dir
        self.last_layer = 'layer.23'
        self.max_length = 512
        self.model, self.tokenizer = get_model_and_tokenizer(model_name, num_labels=num_labels)
        self.freeze_strategy = freeze_strategy
        self.batch_size = batch_size
        self.num_labels = num_labels
        if model_name == 'seBERT':
            self.max_length = 128
        if model_name == 'seBERTfinal':
            self.max_length = 128
        if model_name == 'BERTbase':
            self.last_layer = 'layer.11'

    def fit(self, X, y):
        """Fit is finetuning from the pre-trained model."""

        if self.freeze_strategy == 'last2layer':
            for name, param in self.model.bert.named_parameters():
                if not name.startswith('pooler') and self.last_layer not in name:
                    param.requires_grad = False

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_train_tokens = self.tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=self.max_length)
        X_val_tokens = self.tokenizer(X_val.tolist(), padding=True, truncation=True, max_length=self.max_length)

        train_dataset = Dataset(X_train_tokens, y_train)
        eval_dataset = Dataset(X_val_tokens, y_val)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        training_args = TrainingArguments(
            output_dir                  = self.checkpoints_dir,
            num_train_epochs            = 5,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size  = self.batch_size,
            gradient_accumulation_steps = 4,
            eval_accumulation_steps     = 10,
            evaluation_strategy         = 'epoch', 
            load_best_model_at_end      = True
        )

        self.trainer = Trainer(
            model           = self.model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = eval_dataset,
            compute_metrics = compute_metrics
        )
        if self.num_labels > 2:
            self.trainer.compute_metrics = compute_metrics_multi_label
        
        print(self.trainer.train())
        return self

    def predict_proba(self, X, y=None):
        """This is kept simple intentionally, for larger Datasets this would be too ineficient,
        because we would effectively have a batch size of 1."""
        y_probs = []
        self.trainer.model.eval()
        with torch.no_grad():
            for _, X_row in enumerate(X):
                inputs = self.tokenizer(X_row, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to('cuda')
                outputs = self.trainer.model(**inputs)
                probs = outputs[0].softmax(1).cpu().detach().numpy()
                y_probs.append(probs)
        return y_probs

    def predict(self, X, y=None):
        """Predict is evaluation."""
        y_probs = self.predict_proba(X, y)
        y_pred = []
        for y_prob in y_probs:
            if self.num_labels > 2:
                y_pred.append(y_prob.argmax())
            else:
                y_pred.append(y_prob.argmax() == 1)
        return y_pred

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.model.save_pretrained(path)
