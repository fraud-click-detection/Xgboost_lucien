
# coding: utf-8

# In[1]:


import gc
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt


# In[2]:


train_sample = pd.read_csv('train_sample.csv',nrows=10000000)
train_sample.head()


# In[3]:


train_sample['click_time']=pd.to_datetime(train_sample['click_time'])
train_sample['attributed_time'] = pd.to_datetime(train_sample['attributed_time'])
train_sample['dow']=train_sample['click_time'].dt.dayofweek
train_sample['doy']=train_sample['click_time'].dt.dayofyear
train_sample['click_hour']=train_sample['click_time'].dt.hour


# In[4]:


train_sample.head(100)


# In[5]:


variables = ['ip', 'app', 'device', 'os', 'channel','is_attributed','dow','doy']
for v in variables:
    train_sample[v] = train_sample[v].astype('category')


# In[6]:


train_sample['click_time']=pd.to_datetime(train_sample['click_time'])
train_sample['attributed_time'] = pd.to_datetime(train_sample['attributed_time'])
train_sample['dow']=train_sample['click_time'].dt.dayofweek
train_sample['doy']=train_sample['click_time'].dt.dayofyear
train_sample[['doy','is_attributed']].groupby(['doy'], as_index=True).count().plot()
plt.title('WEEKOFDAY CLICK FREQUENCY');
plt.ylabel('Number of Clicks');

a=train_sample[['dow','is_attributed']].groupby(['dow'], as_index=True).count()
b=a/10000000
b.plot()
plt.title('WEEKOFDAY CONVERSION RATIO');
plt.ylabel('Converted Ratio');

train_sample[['doy','is_attributed']].groupby(['doy'], as_index=True).count().plot()
plt.title('WEEKOFYEAR CLICK FREQUENCY');
plt.ylabel('Number of Clicks');

a=train_sample[['doy','is_attributed']].groupby(['doy'], as_index=True).count()
b=a/10000000
b.plot()
plt.title('WEEKOFYEAR CONVERSION RATIO');
plt.ylabel('Converted Ratio');

