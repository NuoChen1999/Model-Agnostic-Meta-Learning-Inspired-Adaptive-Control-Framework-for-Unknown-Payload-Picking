import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch.nn import functional as F
from datetime import datetime
import random

np.random.seed(4)
num_samples = 1000
X_1 = pd.read_csv('Train_x_1.csv')
Y_1 = pd.read_csv('Train_y_1.csv')
X_2 = pd.read_csv('Train_x_2.csv')
Y_2 = pd.read_csv('Train_y_2.csv')
X_3 = pd.read_csv('Train_x_3.csv')
Y_3 = pd.read_csv('Train_y_3.csv')
X_4 = pd.read_csv('Train_x_4.csv')
Y_4 = pd.read_csv('Train_y_4.csv')
X_5 = pd.read_csv('Train_x_5.csv')
Y_5 = pd.read_csv('Train_y_5.csv')
X_6 = pd.read_csv('Train_x_6.csv')
Y_6 = pd.read_csv('Train_y_6.csv')
X_7 = pd.read_csv('Train_x_7.csv')
Y_7 = pd.read_csv('Train_y_7.csv')
X_8 = pd.read_csv('Train_x_8.csv')
Y_8 = pd.read_csv('Train_y_8.csv')
X_9 = pd.read_csv('Train_x_9.csv')
Y_9 = pd.read_csv('Train_y_9.csv')
X_10 = pd.read_csv('Train_x_10.csv')
Y_10 = pd.read_csv('Train_y_10.csv')
Test_X_1 = pd.read_csv('Test_x_1.csv')
Test_Y_1 = pd.read_csv('Test_y_1.csv')


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, Y_1, test_size=0.2, random_state=4)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, Y_2, test_size=0.2, random_state=4)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, Y_3, test_size=0.2, random_state=4)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4, Y_4, test_size=0.2, random_state=4)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, Y_5, test_size=0.2, random_state=4)
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_6, Y_6, test_size=0.2, random_state=4)
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(X_7, Y_7, test_size=0.2, random_state=4)
X_train_8, X_test_8, y_train_8, y_test_8 = train_test_split(X_8, Y_8, test_size=0.2, random_state=4)
X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(X_9, Y_9, test_size=0.2, random_state=4)
X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(X_10, Y_10, test_size=0.2, random_state=4)

X_train_tensor_1 = torch.tensor(np.array(X_train_1), dtype=torch.float32)
y_train_tensor_1 = torch.tensor(np.array(y_train_1), dtype=torch.float32)
X_test_tensor_1 = torch.tensor(np.array(X_test_1), dtype=torch.float32)
y_test_tensor_1 = torch.tensor(np.array(y_test_1), dtype=torch.float32)
X_train_tensor_2 = torch.tensor(np.array(X_train_2), dtype=torch.float32)
y_train_tensor_2 = torch.tensor(np.array(y_train_2), dtype=torch.float32)
X_test_tensor_2 = torch.tensor(np.array(X_test_2), dtype=torch.float32)
y_test_tensor_2 = torch.tensor(np.array(y_test_2), dtype=torch.float32)
X_train_tensor_3 = torch.tensor(np.array(X_train_3), dtype=torch.float32)
y_train_tensor_3 = torch.tensor(np.array(y_train_3), dtype=torch.float32)
X_test_tensor_3 = torch.tensor(np.array(X_test_3), dtype=torch.float32)
y_test_tensor_3 = torch.tensor(np.array(y_test_3), dtype=torch.float32)
X_train_tensor_4 = torch.tensor(np.array(X_train_4), dtype=torch.float32)
y_train_tensor_4 = torch.tensor(np.array(y_train_4), dtype=torch.float32)
X_test_tensor_4 = torch.tensor(np.array(X_test_4), dtype=torch.float32)
y_test_tensor_4 = torch.tensor(np.array(y_test_4), dtype=torch.float32)
X_train_tensor_5 = torch.tensor(np.array(X_train_5), dtype=torch.float32)
y_train_tensor_5 = torch.tensor(np.array(y_train_5), dtype=torch.float32)
X_test_tensor_5 = torch.tensor(np.array(X_test_5), dtype=torch.float32)
y_test_tensor_5 = torch.tensor(np.array(y_test_5), dtype=torch.float32)
X_train_tensor_6 = torch.tensor(np.array(X_train_6), dtype=torch.float32)
y_train_tensor_6 = torch.tensor(np.array(y_train_6), dtype=torch.float32)
X_test_tensor_6 = torch.tensor(np.array(X_test_6), dtype=torch.float32)
y_test_tensor_6 = torch.tensor(np.array(y_test_6), dtype=torch.float32)
X_train_tensor_7 = torch.tensor(np.array(X_train_7), dtype=torch.float32)
y_train_tensor_7 = torch.tensor(np.array(y_train_7), dtype=torch.float32)
X_test_tensor_7 = torch.tensor(np.array(X_test_7), dtype=torch.float32)
y_test_tensor_7 = torch.tensor(np.array(y_test_7), dtype=torch.float32)
X_train_tensor_8 = torch.tensor(np.array(X_train_8), dtype=torch.float32)
y_train_tensor_8 = torch.tensor(np.array(y_train_8), dtype=torch.float32)
X_test_tensor_8 = torch.tensor(np.array(X_test_8), dtype=torch.float32)
y_test_tensor_8 = torch.tensor(np.array(y_test_8), dtype=torch.float32)
X_train_tensor_9 = torch.tensor(np.array(X_train_9), dtype=torch.float32)
y_train_tensor_9 = torch.tensor(np.array(y_train_9), dtype=torch.float32)
X_test_tensor_9 = torch.tensor(np.array(X_test_9), dtype=torch.float32)
y_test_tensor_9 = torch.tensor(np.array(y_test_9), dtype=torch.float32)
X_train_tensor_10 = torch.tensor(np.array(X_train_10), dtype=torch.float32)
y_train_tensor_10 = torch.tensor(np.array(y_train_10), dtype=torch.float32)
X_test_tensor_10 = torch.tensor(np.array(X_test_10), dtype=torch.float32)
y_test_tensor_10 = torch.tensor(np.array(y_test_10), dtype=torch.float32)

X_test_update_1 = torch.tensor(np.array(Test_X_1), dtype=torch.float32)
Y_test_update_1 = torch.tensor(np.array(Test_Y_1), dtype=torch.float32)

X_train_list = [X_train_tensor_1, X_train_tensor_2, X_train_tensor_3, X_train_tensor_4, X_train_tensor_5, X_train_tensor_6, X_train_tensor_7, X_train_tensor_8, X_train_tensor_9, X_train_tensor_10]
X_test_list = [X_test_tensor_1, X_test_tensor_2, X_test_tensor_3, X_test_tensor_4, X_test_tensor_5, X_test_tensor_6, X_test_tensor_7, X_test_tensor_8, X_test_tensor_9, X_test_tensor_10]
y_train_list = [y_train_tensor_1, y_train_tensor_2, y_train_tensor_3, y_train_tensor_4, y_train_tensor_5, y_train_tensor_6, y_train_tensor_7, y_train_tensor_8, y_train_tensor_9, y_train_tensor_10]
y_test_list = [y_test_tensor_1, y_test_tensor_2, y_test_tensor_3, y_test_tensor_4, y_test_tensor_5, y_test_tensor_6, y_test_tensor_7, y_test_tensor_8, y_test_tensor_9, y_test_tensor_10]

class SimpleNN(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=40):
        super(SimpleNN, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 7)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 7)
        self.relu = nn.ReLU()

    def forward(self, x, params=None):
        if params is None:
            x = x.reshape(x.shape[0], -1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
        else:
            x = x.reshape(x.shape[0], -1)
            x = torch.tanh(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = torch.tanh(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x


model = SimpleNN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# num_epochs = 2000
# batch_size = 32
def train_maml(num_epochs, inner_lr=0.002):
    for epoch in range(num_epochs):
        meta_loss = 0

        for i in random.sample(range(10), 5):  # 5 different tasks in each epoch
            x_train = X_train_list[i]
            y_train = y_train_list[i]
            x_test = X_test_list[i]
            y_test = y_test_list[i]

            pred_y = model(x_train)
            loss = loss_fn(pred_y, y_train)

            # Inner loop update
            gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            fast_weights = {}
            for (name, param), grad in zip(model.named_parameters(), gradients):
                fast_weights[name] = param - inner_lr * grad

            maml_model_temp = SimpleNN()

            # Calculate loss with updated weights
            pred_y_new = maml_model_temp(x_test, params=fast_weights)
            new_loss = loss_fn(pred_y_new, y_test)

            meta_loss += new_loss

        # Meta Update
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {meta_loss.item():.4f}')

train_maml(num_epochs=2000)

y_pred = model(X_test_update_1).detach().numpy()
y_pred_original = y_pred
y_test_original = Test_Y_1

mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)

plt.figure(figsize=(8, 6))
plt.plot(y_test_original.values[:,1], label='real', linestyle='-')
plt.plot(y_pred_original[:,1], label='predict', linestyle='--')
plt.legend()
plt.show()

# # MAML update
# predictions = []
# loss_print = []
# fast_weights = {name: param.clone() for name, param in model.named_parameters()}
# now = datetime.now()
# original_time = now.timestamp()
# for step in range(2000):
#     # test_x_train, test_y_train, _, _ = generate_data(test_task)
#     pred_y = model(X_test_update_1, params=fast_weights)
#     loss = loss_fn(pred_y, Y_test_update_1)
#
#     gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
#
#     for (name, param), grad in zip(fast_weights.items(), gradients):
#         fast_weights[name] = param - 0.002 * grad
#
#     maml_pred = model(X_test_update_1, params=fast_weights)#.detach().numpy()
#     predictions.append(maml_pred)
#
#     now = datetime.now()
#     current_time = now.timestamp()
#
#     print(f"Step {step+1}, Loss:{loss_fn(maml_pred, Y_test_update_1)}, Time:{current_time-original_time}")
#
#     original_time = current_time
#     rmse = loss_fn(maml_pred, Y_test_update_1)
#     loss_print.append(rmse.detach())
#
# # Plot result
# for i, pred in enumerate(predictions):
#     if i in (0,9,99,999,1999):
#         plt.plot(pred.detach().numpy()[:,6], label=f"Step {i+1}", linewidth=0.5)
#
# plt.plot(Y_test_update_1.detach().numpy()[:,6], label="True Function", color="black", linewidth=0.5)
# plt.legend()
# plt.title("MAML")
# plt.xlabel("x")
# # plt.ylim(-1,1)
# plt.ylabel("y")
# plt.show()

# # Plot loss
# plt.figure(figsize=(10, 5))
# plt.plot(loss_print)
# plt.title("Loss")
# plt.show()
