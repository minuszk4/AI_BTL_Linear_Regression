import pandas as pd
import numpy as np
import logging
import joblib
import json


class Linear_Regression:
    def __init__(self, learning_rate=0.001, no_of_itr=1000):
        self.learning_rate = learning_rate
        self.no_of_itr = no_of_itr
        self.w = None
        self.b = None

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = np.array(X)
        self.Y = np.array(Y)
        print(X.shape,Y.shape,self.w.shape)
        logging.info(f"Starting training with {self.no_of_itr} iterations, learning rate: {self.learning_rate}")
        for i in range(self.no_of_itr):
            self.update_weights()
            if i % 100 == 0 or i == self.no_of_itr - 1:
                y_pred = self.predict(self.X)
                mse = self.mean_squared_error(self.Y, y_pred)
                logging.info(f"Iteration {i}, Mean Squared Error: {mse}")

        logging.info("Training completed")
        return self

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        dw = - (2 / self.m) * np.dot(self.X.T, (self.Y - Y_prediction))
        db = - (2 / self.m) * np.sum(self.Y - Y_prediction)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        # print(self.b)

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        mean_y = np.mean(y_true)
        ss_total = np.sum((y_true - mean_y) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    def MAE(self,y_true,y_pred):
        return np.mean(np.abs(y_true-y_pred))
    def print_weights(self):
        print('Weights for the respective features are:')
        print(self.w)
        print('Bias value for the regression is:', self.b)
    def save_model(self, file_path):
        joblib.dump({
            'w': self.w,
            'b': self.b,
            'mean': getattr(self, 'mean', None),
            'std': getattr(self, 'std', None)
        }, file_path)

    def load_model(self, file_path):
        params = joblib.load(file_path)
        self.w = params['w']
        self.b = params['b']
        self.mean = params.get('mean', None)
        self.std = params.get('std', None)

def standardize_fit_transform(X):
    X = np.array(X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std

def standardize_transform(X, mean, std):
    X = np.array(X)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std

def train_test_split(X, y, train_size=0.8, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(train_size * n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=[ 'id', 'month'])
    df['balcony'] = df['balcony'].fillna('none')
    df['direction'] = df['direction'].fillna('unknown')
    df['ward'] = df['ward'].fillna('unknown')
    df['investor'] = df['investor'].fillna('unknown')
    df['project'] = df['project'].fillna('unknown')
    unique_investors = df['investor'].unique().tolist()
    with open("investors.json", "w", encoding="utf-8") as f:
        json.dump(unique_investors, f, ensure_ascii=False, indent=2)
    unique_projects = df['project'].unique().tolist()
    # print(unique_investors)
    with open("projects.json", "w", encoding="utf-8") as f:
        json.dump(unique_projects, f, ensure_ascii=False, indent=2)
    return df

def one_hot_encode(df, columns, drop_first=True):
    df_encoded = df.copy()
    encoded_columns = []
    for col in columns:
        unique_values = df[col].unique()
        if drop_first:
            unique_values = unique_values[1:]

        for val in unique_values:
            new_col = (df[col] == val).astype(int)
            new_col.name = f"{col}_{val}"
            encoded_columns.append(new_col)
    # print(df_encoded)
    df_encoded = df.drop(columns=columns)
    df_encoded = pd.concat([df_encoded] + encoded_columns, axis=1)
    df_encoded = df_encoded.copy()
    return df_encoded

def main():
    file_path = "/kaggle/input/chung-cu-hanoi/dataset.csv"  
    df = load_data(file_path)
    # print(df)
    df_encoded = df.copy()
    unique_wards = df_encoded['ward'].unique()
    print(unique_wards)
    print(df_encoded['direction'].unique().shape)
    categorical_cols = ['investor','direction', 'balcony', 'district', 'ward','project']
    df_encoded = one_hot_encode(df_encoded, categorical_cols, drop_first=False) 
    print(df_encoded.shape)
    columns = df_encoded.drop(columns=['price']).columns
    X = df_encoded.drop(columns=['price']).values
    y = df_encoded['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    X_train, mean, std = standardize_fit_transform(X_train)
    X_test = standardize_transform(X_test, mean, std)

    model = Linear_Regression(learning_rate=0.1, no_of_itr=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = model.mean_squared_error(y_test, y_pred)
    mae = model.MAE(y_test,y_pred)
    r2 = model.r2_score(y_test, y_pred)
    model.mean = mean
    model.std = std
    model.save_model('linear_regression_model.pkl')
    print(f"final Mean Squared Error: {mse}")
    print(f"final Mean Absolute Error: {mae}")
    print(f"final R-squared: {r2}")
    model.print_weights()
    
    
    print("\nSo sánh y_test và y_pred:")
    comparison_df = pd.DataFrame({
        'y_test (Thực tế)': y_test,
        'y_pred (Dự đoán)': y_pred
    })
    print(comparison_df)

if __name__ == "__main__":
    main()