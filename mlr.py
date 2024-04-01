# # Custom Multiple Linear Regression
# We're out to customize the loss function, so we can't just rely on the closed-form solution of linear regression. The loss function is still differentiable with respect to each model parameter, so we're going to rely on autodifferentiation to implement stochastic gradient descent.

# %%
import pandas as pd
import jax.numpy as np
import numpy
from jax import grad, jit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

# %%
df = pd.read_csv('Census_Micro.csv')
df = df.iloc[:10000]

# %%
X = OneHotEncoder(sparse_output=False).fit_transform(df[['SEX', 'JWTRNS', 'COW', 'SCHL']].to_numpy())
X = numpy.concatenate([X, df[['AGEP', 'PWGTP']].to_numpy()], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, numpy.array(df['PERNP']))

# %%
class FairLinearRegression:
    coefs: np.ndarray
    bias: np.ndarray

    def __init__(self, X, y, separation_weight=0, A_col=0) -> None:
        self.coefs = numpy.random.normal(size=(X.shape[1], ))
        self.separation_weight = separation_weight
        self.A_col = A_col
        self.bias = 0.0
        self.jit_loss = jit(self.lr_loss)
        self.grad = grad(self.lr_loss, argnums=[0, 2])

        self.y_buckets = numpy.unique(numpy.maximum(y // 10000, 0))

    def predict(self, X):
        return np.matmul(X, self.coefs) + self.bias

    def lr_loss(self, W, X, bias, y):
        pred = np.matmul(X, W) + bias
        return np.mean((y - pred)**2) + self.separation_weight * self.separation(pred, y, X, self.A_col)

    def train_iterate(self, X, y, learning_rate=0.001):
        rows = numpy.random.choice(X.shape[0], int(X.shape[0] * 0.1) + 1)
        X_batch = X[rows, :]
        W_grad, bias_grad = self.grad(self.coefs, X_batch, self.bias, y[rows])
        self.coefs -= W_grad * learning_rate
        self.bias -= bias_grad * learning_rate
    
    def train(self, X, y, learning_rate=0.001):
        loss = self.lr_loss(self.coefs, X, self.bias, y)
        
        i = 0
        while True:
            if i % 100 == 0:
                print(i, loss)
            self.train_iterate(X, y, learning_rate)
            new_loss = self.lr_loss(self.coefs, X, self.bias, y)
            if new_loss > loss or abs(new_loss - loss) < 1e-2:
                return
            loss = new_loss
            i += 1

    def separation(self, pred, act, X, A_col):
        yhat_test_buckets = np.maximum(pred // 10000, 0)
        y_test_buckets = np.maximum(act // 10000, 0)

        prob_diff_sum = np.array(0)
        prob_diff_count = 0
        for yhat_bucket in self.y_buckets:
            for y_bucket in self.y_buckets:
                # P(R | Y, A) = P(R, Y, A) / P(Y, A)

                y_a = (X[:, A_col] == 0) & \
                    (y_test_buckets == y_bucket)
                r_y_a = y_a & (yhat_test_buckets == yhat_bucket)
                prob_a = numpy.sum(r_y_a) / numpy.sum(y_a)

                
                y_a = (X[:, A_col] == 1) & \
                    (y_test_buckets == y_bucket)
                r_y_a = y_a & (yhat_test_buckets == yhat_bucket)
                prob_b = numpy.sum(r_y_a) / numpy.sum(y_a)
                
                prob_diff_sum += 0 if np.isnan(prob_a - prob_b) else abs(prob_a - prob_b)
                prob_diff_count += 1
        return prob_diff_sum / prob_diff_count if prob_diff_count != 0 else 0

flr = FairLinearRegression(X_train, y_train, separation_weight=200000)

# %%
nn = MLPRegressor(hidden_layer_sizes=(80,))
nn.fit(X_train, y_train)
lr = LinearRegression()
lr.fit(X_train, y_train)

# %%
print("least squares NN", numpy.mean((nn.predict(X_test) - y_test)**2))
print("least squares LR", numpy.mean((lr.predict(X_test) - y_test)**2))

# %%
for i in range(1,61):
    flr.train_iterate(X_train, y_train, 2e-5)
    if i % 10 == 0:
        print(i, flr.lr_loss(flr.coefs, X_train, flr.bias, y_train))

# %%
# what is the loss without the separation term?
spw = flr.separation_weight
flr.separation_weight = 0
flr.lr_loss(flr.coefs, X_test, flr.bias, y_test)
flr.separation_weight = spw

# %%
print("separation NN", flr.separation(nn.predict(X_test), y_test, X_test, 0))
print("separation LR", flr.separation(lr.predict(X_test), y_test, X_test, 0))
print("separation FLR", flr.separation(flr.predict(X_test), y_test, X_test, 0))

# This is independence with respect to sex
# E[Yhat | Sex = Male], E[Yhat | Sex = Female]
# (numpy.mean(yhat_test[X_test[:, 0] == 0]).item(), numpy.mean(yhat_test[X_test[:, 0] == 1]).item())