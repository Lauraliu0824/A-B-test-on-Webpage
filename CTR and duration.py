#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(42)


# In[2]:


df = pd.read_csv('course_page_actions.csv')
df.head()


# In[3]:


# Get dataframe with all records from control group
control_df = df.query('group == "control"')

# Compute click through rate for control group
control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()

# Display click through rate
control_ctr


# # CTR

# In[4]:


# Get dataframe with all records from experiment group
experiment_df = df.query('group == "experiment"')

# Compute click through rate for experiment group
experiment_ctr = experiment_df.query('action == "enroll"').id.nunique() / experiment_df.query('action == "view"').id.nunique()


# Display click through rate
experiment_ctr


# In[5]:


# Compute the observed difference in click through rates
obs_diff = experiment_ctr -control_ctr

# Display observed difference
obs_diff


# In[6]:


# Create a sampling distribution of the difference in proportions
# with bootstrapping
diffs = []
size = df.shape[0]
for _ in range(10000):
    b_samp = df.sample(size, replace=True)
    control_df = b_samp.query('group == "control"')
    experiment_df = b_samp.query('group == "experiment"')
    control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()
    experiment_ctr = experiment_df.query('action == "enroll"').id.nunique() / experiment_df.query('action == "view"').id.nunique()
    diffs.append(experiment_ctr - control_ctr)


# In[7]:


# Convert to numpy array
diffs = np.array(diffs)

# Plot sampling distribution
plt.hist(diffs)


# In[8]:


# Simulate distribution under the null hypothesis
null_vals = np.random.normal(0,diffs.std(),diffs.size)

# Plot the null distribution
plt.hist(null_vals)


# In[9]:


# Plot observed statistic with the null distibution
plt.hist(null_vals);
plt.axvline(x=obs_diff,color='red');


# In[10]:


# Compute p-value
(null_vals > obs_diff).mean()


# # Duration

# In[9]:


views = df.query('action == "view"')
reading_tiems = views.groupby(['id','group'])['duration'].mean().reset_index()


# In[12]:


control_mean = df.query('group == "control"')['duration'].mean()
experiment_mean = df.query('group == "experiment"')['duration'].mean()
control_mean, experiment_mean


# In[13]:


obs_diff = experiment_mean-control_mean
obs_diff


# In[20]:


diffs = []
for _ in range(10000):
    b_samp = df.sample(df.shape[0], replace=True)
    control_mean = b_samp.query('group == "control"')['duration'].mean()
    experiment_mean = b_samp.query('group == "experiment"')['duration'].mean()
    diffs.append(experiment_mean - control_mean)


# In[21]:


diffs = np.array(diffs)
plt.hist(diffs)


# In[22]:


null_vals = np.random.normal(0, diffs.std(),diffs.size)
plt.hist(null_vals);
plt.axvline(x=obs_diff, color = "red")


# In[23]:


# Compute p-value
(null_vals > obs_diff).mean()


# In[ ]:




