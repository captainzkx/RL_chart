from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

MAAC_reward_data  = pd.read_csv('MAAC_reward.csv')
MAAC_reward = MAAC_reward_data.iloc[:,1:].values
date_buff=MAAC_reward.reshape(50000)

MADDPG_reward_data  = pd.read_csv('MADDPG_reward.csv')
MADDPG_reward = MADDPG_reward_data.iloc[:,1:].values
date_buff_1=MADDPG_reward.reshape(50000)

""" b, a = signal.butter(8, 0.01, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
date_buff=images.reshape(50000)
filtedData = signal.filtfilt(b, a,date_buff )  #data为要过滤的信号 """

filtedData_MAAC = signal.savgol_filter(date_buff,2001,3)
filtedData_MADDPG = signal.savgol_filter(date_buff_1,2001,3)
plt.ylabel("Reward") # x轴名称
plt.xlabel("Episode") # y 轴名称

plt.plot(filtedData_MAAC, color="#1E90FF")
plt.plot(filtedData_MADDPG, color="#FF7F00"	)
plt.plot(MADDPG_reward, color="#90EE90"	,alpha=0.3)
plt.plot(MAAC_reward, color="#56cbf5",alpha=0.3)
plt.legend(["MAAC", "MADDPG"], loc='lower right') 
plt.show()