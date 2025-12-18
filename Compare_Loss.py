import pandas as pd
import matplotlib.pyplot as plt

Loss1_data = 'Loss_DAIML.csv'
Loss2_data = 'Loss_Proposed.csv'
Loss1 = pd.read_csv(Loss1_data)
Loss2 = pd.read_csv(Loss2_data)

plt.grid()
plt.plot(Loss1, label='DAIML [11]')
plt.plot(Loss2, label='Proposed Algorithm')
plt.xlabel('Time Step', fontsize='15')
plt.ylabel('Loss', fontsize='15')
plt.xticks([0, 1000, 2000, 3000], fontsize='10')
plt.yticks([0, 50, 100, 150], fontsize='10')
plt.legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize='15')
plt.show()