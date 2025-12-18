import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from datetime import datetime

# Generate task y = asin(x) + bcos(x)
def generate_task():
    a = np.random.uniform(0.1, 5)
    b = np.random.uniform(0, np.pi)
    return lambda x: a * np.sin(x) + b * np.cos(x), (a, b)

# Generate data point based on task
def generate_data(task, num_points=50):
    x_train = np.random.uniform(-5, 5, size=(num_points, 1))
    y_train = task(x_train)
    x_test = np.random.uniform(-5, 5, size=(num_points, 1))
    y_test = task(x_test)
    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Compute error
def compute_rmse(pred, true):
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse

# Define neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, params=None):
        if params is None:
            x = x.view(-1, 1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
        else:
            x = x.view(-1, 1)
            x = torch.tanh(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = torch.tanh(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x

# Initialize neural network and loss function
model_nn = SimpleNN()
model_maml = SimpleNN()
optimizer_nn = optim.Adam(model_nn.parameters(), lr=0.01)
optimizer_maml = optim.Adam(model_maml.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# Train traditional neural network
def train_nn(num_epochs=1000):
    for epoch in range(num_epochs):
        task, _ = generate_task()
        x, y, _, _ = generate_data(task)

        pred_y = model_nn(x)
        loss = loss_fn(pred_y, y)

        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()


# Train MAML
def train_maml(num_epochs=1000, inner_lr=0.002):
    for epoch in range(num_epochs):
        meta_loss = 0

        for _ in range(10):  # 10 different tasks in each epoch
            task, _ = generate_task()
            x_train, y_train, x_test, y_test = generate_data(task)

            pred_y = model_maml(x_train)
            loss = loss_fn(pred_y, y_train)

            # Inner loop update
            gradients = torch.autograd.grad(loss, model_maml.parameters(), create_graph=True)

            fast_weights = {}
            for (name, param), grad in zip(model_maml.named_parameters(), gradients):
                fast_weights[name] = param - inner_lr * grad

            maml_model_temp = SimpleNN()
            pred_y_new = maml_model_temp(x_test, params=fast_weights)
            new_loss = loss_fn(pred_y_new, y_test)

            meta_loss += new_loss

        # Meta Update
        optimizer_maml.zero_grad()
        meta_loss.backward()
        optimizer_maml.step()


# Train traditional neural network and MAML
train_nn(num_epochs=1000)
train_maml(num_epochs=1000)

# Generate test task
test_task, test_params = generate_task()
test_x = torch.linspace(-5, 5, 1000).unsqueeze(1)
test_y = test_task(test_x.numpy())

# Traditional neural network prediction
nn_pred = model_nn(test_x).detach().numpy()

# MAML prediction
maml_pred_init = model_maml(test_x).detach().numpy()

predictions = []
loss_print = []

now = datetime.now()
original_time = now.timestamp()

# MAML update
test_x_train, test_y_train, _, _ = generate_data(test_task)
updated_model = SimpleNN()
updated_model.load_state_dict(model_maml.state_dict())
for step in range(100):
    pred_y = updated_model(test_x_train)
    loss = loss_fn(pred_y, test_y_train)

    grads = torch.autograd.grad(loss, updated_model.parameters(), create_graph=True)

    with torch.no_grad():
        for param, grad in zip(updated_model.parameters(), grads):
            param.copy_(param - 0.002 * grad)  # θ' = θ - α * ∇L

    maml_pred = updated_model(test_x).detach().numpy()
    predictions.append(maml_pred)

    now = datetime.now()
    current_time = now.timestamp()

    print(f"Step {step+1}, Loss:{compute_rmse(maml_pred, test_y)}, Time:{current_time-original_time}")

    original_time = current_time
    mse, rmse = compute_rmse(maml_pred, test_y)
    loss_print.append(rmse)


# Plot result
for i, pred in enumerate(predictions):
    if i in (0,9,19,49,99):
        plt.plot(test_x.numpy(), pred, label=f"Step {i+1}")

mse_nn, rmse_nn = compute_rmse(nn_pred, test_y)
mse_maml_init, rmse_maml_init = compute_rmse(maml_pred_init, test_y)

print("Mean Square Error (MSE) and Root Mean Square Error (RMSE):")
print(f"NN:         MSE = {mse_nn:.5f}, RMSE = {rmse_nn:.5f}")
print(f"MAML Init:  MSE = {mse_maml_init:.5f}, RMSE = {rmse_maml_init:.5f}")

plt.plot(test_x.numpy(), test_y, label="True Function", color="black", linewidth=2)
plt.plot(test_x.numpy(), nn_pred, label="NN Prediction", linestyle="dashed", color="blue")
plt.plot(test_x.numpy(), maml_pred_init, label="MAML Init", linestyle="dashed", color="red")
plt.scatter(test_x_train.numpy(), test_y_train.numpy(), label="Used for grad", color="purple", marker="x")
plt.legend()
plt.title("MAML vs NN: Adaptation Comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(loss_print)
plt.title("Loss")
plt.legend()
plt.show()