import pandas as pd
import numpy as np
import random
class MyLineReg():

    def __init__(self, weights=None, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_score = None

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, reg={self.reg}, l1_coef={self.l1_coef}, l2_coef={self.l2_coef}, sgd_sample={self.sgd_sample}, random_state={self.random_state}'

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, reg={self.reg}, l1_coef={self.l1_coef}, l2_coef={self.l2_coef}, sgd_sample={self.sgd_sample}, random_state={self.random_state}'

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X_with_intercept = pd.concat([pd.Series(1, index=X.index, name='w0'), X], axis=1)
        y_pr = X_with_intercept @ self.weights
        return y_pr
    
    def fit(self, X, y, verbose=False):
        X_with_intercept = pd.concat([pd.Series(1, index=X.index, name='w0'), X], axis=1)
        self.weights = np.ones(X_with_intercept.shape[1])
        random.seed(self.random_state)

        for i in range(1, self.n_iter + 1):
            learning_rate = self.learning_rate if not callable(self.learning_rate) else self.learning_rate(i)
            
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, int):
                    sample_size = min(self.sgd_sample, len(X))
                elif isinstance(self.sgd_sample, float):
                    sample_size = int(self.sgd_sample * len(X))
                else:
                    raise ValueError("Invalid type for sgd_sample. Should be int or float.")
                
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
                X_sampled = X_with_intercept.iloc[sample_rows_idx]
                y_sampled = y.iloc[sample_rows_idx]
            else:
                X_sampled = X_with_intercept
                y_sampled = y
            
            y_pr = X_sampled @ self.weights
            mse = ((y_pr - y_sampled)**2).sum() / len(y_sampled)

            if self.reg == 'l1':
                l1_reg = self.l1_coef * np.sign(self.weights)
                loss = mse + np.sum(np.abs(l1_reg))
            elif self.reg == 'l2':
                l2_reg = self.l2_coef * self.weights
                loss = mse + np.sum(np.square(l2_reg))
            elif self.reg == 'elasticnet':
                l1_reg = self.l1_coef * np.sign(self.weights)
                l2_reg = self.l2_coef * self.weights
                loss = mse + np.sum(np.abs(l1_reg)) + np.sum(np.square(l2_reg))
            else:
                loss = mse

            if self.metric is not None:
                if self.metric == 'mae':
                    metric_val = (np.abs(y_pr - y)).mean()
                elif self.metric == 'mse':
                    metric_val = ((y_pr - y) ** 2).mean()
                elif self.metric == 'rmse':
                    metric_val = np.sqrt(((y_pr - y) ** 2).mean())
                elif self.metric == 'mape':
                    metric_val = (np.abs((y - y_pr) / y)).mean() * 100
                elif self.metric == 'r2':
                    ss_res = ((y - y_pr) ** 2).sum()
                    ss_tot = ((y - y.mean()) ** 2).sum()
                    metric_val = 1 - (ss_res / ss_tot)

            grad = (2 / len(y_sampled)) * (y_pr - y_sampled) @ X_sampled
            if self.reg == 'l1':
                grad += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                grad += 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                grad += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            self.weights = self.weights - learning_rate * grad

            if verbose and i % verbose == 0:
                print(f'{i} | loss: {loss} | {self.metric}: {metric_val} | learning_rate: {learning_rate}')

        if self.metric is not None:
            self.best_score = metric_val

    def get_best_score(self):
        return round(self.best_score, 10)