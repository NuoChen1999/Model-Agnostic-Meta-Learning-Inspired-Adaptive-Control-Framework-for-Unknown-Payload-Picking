import numpy as np
import pandas as pd

train_x_1 = pd.read_csv('Train_x_1.csv')
train_x_2 = pd.read_csv('Train_x_2.csv')
train_x_3 = pd.read_csv('Train_x_3.csv')
train_x_4 = pd.read_csv('Train_x_4.csv')
train_x_5 = pd.read_csv('Train_x_5.csv')
train_x_6 = pd.read_csv('Train_x_6.csv')
train_x_7 = pd.read_csv('Train_x_7.csv')
train_x_8 = pd.read_csv('Train_x_8.csv')
train_x_9 = pd.read_csv('Train_x_9.csv')
train_x_10 = pd.read_csv('Train_x_10.csv')

train_y_1 = pd.read_csv('Train_y_1.csv')
train_y_2 = pd.read_csv('Train_y_2.csv')
train_y_3 = pd.read_csv('Train_y_3.csv')
train_y_4 = pd.read_csv('Train_y_4.csv')
train_y_5 = pd.read_csv('Train_y_5.csv')
train_y_6 = pd.read_csv('Train_y_6.csv')
train_y_7 = pd.read_csv('Train_y_7.csv')
train_y_8 = pd.read_csv('Train_y_8.csv')
train_y_9 = pd.read_csv('Train_y_9.csv')
train_y_10 = pd.read_csv('Train_y_10.csv')

merged_x = np.vstack((train_x_1, train_x_2, train_x_3, train_x_4, train_x_5, train_x_6, train_x_7, train_x_8, train_x_9, train_x_10))
merged_y = np.vstack((train_y_1, train_y_2, train_y_3, train_y_4, train_y_5, train_y_6, train_y_7, train_y_8, train_y_9, train_y_10))

np.savetxt('Train_x_merge.csv', merged_x, delimiter=',', comments='', fmt='%.5f')
np.savetxt('Train_y_merge.csv', merged_y, delimiter=',', comments='', fmt='%.5f')

