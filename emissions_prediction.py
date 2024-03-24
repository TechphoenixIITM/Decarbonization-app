# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# Sample data for demonstration purposes (replace with your actual data)
# X represents input features (e.g., production parameters)
# y represents emissions data
X = np.array([[100, 200], [150, 250], [200, 300], [250, 350]])  # Example input features (e.g., production parameters)
y = np.array([50, 60, 70, 80])  # Example emissions data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors for TensorFlow and PyTorch
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

X_train_pt = torch.tensor(X_train, dtype=torch.float32)
y_train_pt = torch.tensor(y_train, dtype=torch.float32)
X_test_pt = torch.tensor(X_test, dtype=torch.float32)
y_test_pt = torch.tensor(y_test, dtype=torch.float32)

# TensorFlow Implementation
def train_tf_model(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

# PyTorch Implementation
class Net(nn.Module):
    def _init_(self, input_size, hidden_size1, hidden_size2):
        super(Net, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_pt_model(X_train, y_train, X_test, y_test):
    model = Net(input_size=X_train.shape[1], hidden_size1=64, hidden_size2=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
    return model

# Train TensorFlow model
tf_model = train_tf_model(X_train_tf, y_train_tf, X_test_tf, y_test_tf)

# Train PyTorch model
pt_model = train_pt_model(X_train_pt, y_train_pt, X_test_pt, y_test_pt)

# Make predictions and evaluate models
y_pred_tf = tf_model.predict(X_test_tf).flatten()
y_pred_pt = pt_model(X_test_pt).detach().numpy().flatten()

mse_tf = mean_squared_error(y_test, y_pred_tf)
mse_pt = mean_squared_error(y_test, y_pred_pt)

print("Mean Squared Error (TensorFlow):", mse_tf)
print("Mean Squared Error (PyTorch):", mse_pt)