# generate and train SVC model for audio sentiment detection


import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emotion import module_dir, root_dir

DATA_DIR = Path(root_dir / "data/processed/audio")
FEATURES = Path(DATA_DIR / "audio_features.csv")
LABELS = Path(DATA_DIR / "sentiment_labels.csv")
ARTIFACTS_DIR = Path(module_dir / "artifacts")

def create_datasets(in_features, in_labels, test_size=0.2):
    '''
        - create aligned datasets with features and labels
        - split into train/test
        - fit StandardScaler on train features
        - scale train/test features
        in_features : pd.DataFrame containing audio clip features
                      (mean of each audio clips 40 mfcc's)
        in_labels   : sentiment labels for corresponding features
        test_size   : test data size
        returns     : X_train, X_test, y_train, y_test, scaler

    '''
    labels = in_labels.copy()
    features = in_features.copy().reindex(labels.index)
    X_train, X_test, y_train, y_test = \
        train_test_split(features, labels, test_size=test_size,
                         random_state=101,
                         stratify = labels.values.argmax(axis=1))
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("\ntraining label count:")
    print(y_train.sum(axis=0))
    print("\ntest label count:")
    print(y_test.sum(axis=0))
    
    return X_train, X_test, y_train, y_test, scaler

def calc_metrics_per_class(y_true, y_pred, class_names=None):
    '''
        - calculate metrics per class (precision, recall, f1_score)
        - plus macro avergages
        y_true  : true labels
        y_pred  : predicted labels
        classes : class names

    '''
    if len(y_true.shape) > 1:
        conf_mtx = pd.DataFrame(
                            confusion_matrix(
                                y_true.values.argmax(axis=1),
                                y_pred.argmax(axis=1)
                            )
                        )
    else:
        conf_mtx = pd.DataFrame(
                            confusion_matrix(
                                y_true,
                                y_pred
                            )
                        )

    if class_names != None:
        conf_mtx.index = class_names
        conf_mtx.columns = class_names
    else:
        class_names = sorted(list(y_true.unique()))
        conf_mtx.index = class_names
        conf_mtx.columns = class_names

    class_metrics = {}
    for c in class_names:
        metrics = {}
        metrics['precision'] = \
            round(conf_mtx.loc[c, c] / conf_mtx.loc[:, c].sum(), 3)
        metrics['recall'] = \
            round(conf_mtx.loc[c, c] / conf_mtx.loc[c, :].sum(), 3)
        metrics['f1'] = \
            round(2 * (metrics['precision'] * metrics['recall'])/
                  (metrics['precision'] + metrics['recall']), 3)
        class_metrics[c] = metrics
    class_metrics = pd.DataFrame(class_metrics)
    macro_metrics = class_metrics.sum(axis=1) / 3
    class_metrics = class_metrics.T
    class_metrics.loc['macro'] = macro_metrics.round(3)
    return conf_mtx, class_metrics

def predict_show_metrics(model, X, y, show_confu=False, data_name='data'):
    '''
        predict class, calculate and show metrics
        model  : model
        X  : features matrix
        y  : labels matrix
        show_confu : display confusion matrix
        data_name  : name of data
    '''
    pred = model.predict(X)
    print(f"\n{data_name} accuracy : ",
          round(accuracy_score(y.values.argmax(axis=1), pred),3))

    conf_mtx, metrics = \
    calc_metrics_per_class(y.values.argmax(axis=1), pred,
                           class_names=y.columns.tolist())
    if show_confu:
        print("")
        print(conf_mtx)
    print("")
    print(metrics)

def train_svc(X_train, y_train, C=5):
    '''
        train SVM audio sentiment classifier
        X_train : training features
        y_train : training labels
        C       : regularization parameter
        returns : model
    '''
    print("\nTraining svc audio model ...")
    gamma='auto'
    svc = SVC(C=C, kernel='rbf', gamma=gamma, random_state=101)
    svc.fit(X_train, y_train.values.argmax(axis=1))
    return svc

def main():
    '''
        - load features/labels for model training
        - create train/test datasets
        - train model
        - save model
        - show train/test metrics
    '''
    if Path.is_file(FEATURES) and Path.is_file(LABELS):
        features = pd.read_csv(FEATURES, index_col = 0, header = 0)
        labels = pd.read_csv(LABELS, index_col = 0, header = 0)
        class_names = {c : v for c, v in enumerate(labels.columns.tolist())}
        X_train, X_test, y_train, y_test, scaler = \
            create_datasets(features, labels)

        audio_model = train_svc(X_train, y_train)
        audio_model.scaler = scaler
        audio_model.class_names = class_names
        predict_show_metrics(audio_model, X_train, y_train,
                data_name = "Train")

        predict_show_metrics(audio_model, X_test, y_test,
                data_name = "Test")
 
        with open(f"{ARTIFACTS_DIR}/audio_model.pkl", "wb") as f:
            pickle.dump(audio_model, f)
    else:
        print("Either features file or labels file missing from:\n",
            DATA_DIR
        )

if __name__ == "__main__":
    main()
