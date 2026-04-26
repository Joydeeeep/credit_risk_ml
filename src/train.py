from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=3,  # handle imbalance
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    return model

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    return model, scaler