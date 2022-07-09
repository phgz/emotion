# generate and train SVC model for audio sentiment detection


import pickle
from pathlib import Path

import pandas as pd
# from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emotion import module_dir, root_dir

DATA_DIR = Path(root_dir / "data/processed/audio")
FEATURES = Path(DATA_DIR / "mfcc40_3sec_mean_features.csv")
LABELS = Path(DATA_DIR / "sentiment_labels.csv")
ARTIFACTS_DIR = Path(module_dir / "artifacts")

def create_datasets(in_features, in_labels, test_size=0.2):
    labels = in_labels.copy()
    features = in_features.copy().reindex(labels.index)
    X_train, X_test, y_train, y_test = \
        train_test_split(features, labels, test_size=test_size,
                         random_state=100,
                         stratify = labels.values.argmax(axis=1))
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print("\ntraining label count:")
    print(y_train.sum(axis=0))
    print("\ntest label count:")
    print(y_test.sum(axis=0))
    
    return X_train, X_test, y_train, y_test, scaler

def calc_metrics_per_class(y_true, y_pred, classes=None):
    
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

    if classes != None:
        conf_mtx.index = classes
        conf_mtx.columns = classes
    else:
        classes = sorted(list(y_true.unique()))
        conf_mtx.index = classes
        conf_mtx.columns = classes

    class_metrics = {}
    for c in classes:
        metrics = {}
        metrics['precision'] = \
            round(conf_mtx.loc[c, c] / conf_mtx.loc[:, c].sum(), 3)
        metrics['recall'] = \
            round(conf_mtx.loc[c, c] / conf_mtx.loc[c, :].sum(), 3)
        metrics['f1'] = \
            round(2 * (metrics['precision'] * metrics['recall'])/
                  (metrics['precision'] + metrics['recall']), 3)
        class_metrics[c] = metrics
        # metrics
    class_metrics = pd.DataFrame(class_metrics)
    macro_metrics = class_metrics.sum(axis=1) / 3
    class_metrics = class_metrics.T
    class_metrics.loc['macro'] = macro_metrics.round(3)
    return conf_mtx, class_metrics

def predict_show_metrics(model, X, y, show_confu=False, data_name='data'):
    pred = model.predict(X)
    print(f"\n{data_name} accuracy : ",
          round(accuracy_score(y.values.argmax(axis=1), pred),3))

    conf_mtx, metrics = \
    calc_metrics_per_class(y.values.argmax(axis=1), pred,
                           classes=y.columns.tolist())
    if show_confu:
        print("")
        print(conf_mtx)
    print("")
    print(metrics)

def train_svc(X_train, y_train, C=5):
    print("\nTraining svc audio model ...")
    gamma='auto'
    svc = SVC(C=C, kernel='rbf', gamma=gamma, random_state=101)
    svc.fit(X_train, y_train.values.argmax(axis=1))
    return svc

def main():
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
