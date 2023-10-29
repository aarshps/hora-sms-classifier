import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate some training data
X_train = np.random.randn(100, 1)
y_train = (X_train > 0).astype(np.int64)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make a prediction
X_test = np.random.randn(1, 1)
y_pred = model.predict_proba(X_test)[:, 1]

# Print the prediction
print(y_pred)
