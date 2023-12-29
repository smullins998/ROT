#!/usr/bin/env python
# coding: utf-8

# # Load Data and Create Charts

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
import numpy as np
import os
import warnings
from tqdm import tqdm
import random
warnings.filterwarnings("ignore")

print('Loading Data...')

df = pd.read_csv('FullOptions.csv')
print('Data Loaded')

# In[41]:


pd.set_option('display.max_columns', None) 
df.columns = [i.replace('[','').replace(']','').replace(' ','') for i in df.columns]

df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])


# - Convert select columns to float

# In[42]:


col_float = list(df.columns[8:-1])
for i in col_float:
    df[i] = pd.to_numeric(df[i], errors='coerce')


# - Find unique options

# In[44]:

df.sort_values(by='QUOTE_DATE', inplace=True)
df['unique_id'] = [i+'-'+str(j)+'-'+str(k) for i,j,k in zip(df.EXPIRE_DATE,df.STRIKE, df.equity)]
#Let's cut down the dataframe so we can process and filter well


# - Generate graph function

# In[23]:
print('Transformations Done')


import matplotlib.pyplot as plt
import os
import PIL

def generate_plot(temp, name, quotedate):
    fig, ax1 = plt.subplots(figsize=(.8311688311688312*2, .8311688311688312*2))

    ax1.set_xlabel('')
    ax1.plot(temp['QUOTE_DATE'], temp['C_BID'], linewidth=2, color='black')

#     # Create a secondary y-axis for 'C_THETA_pct'
#     ax2 = ax1.twinx()
#     color = 'green'
#     ax2.plot(temp['QUOTE_DATE'], temp['C_THETA_pct'], linewidth=2, color=color, alpha=.7)
#     ax2.tick_params(axis='y', labelcolor=color)

#     # Create a third y-axis for another variable (replace 'C_ANOTHER_pct' with your variable)
#     ax3 = ax1.twinx()
#     ax3.spines['right'].set_position(('outward', 60))  # Adjust the position of the third y-axis
#     color = 'red'
#     ax3.plot(temp['QUOTE_DATE'], temp['C_IV_pct'], linewidth=2, color=color, alpha=.7)
#     ax3.tick_params(axis='y', labelcolor=color)

    # Hide x-axis ticks and labels
    ax1.set_xticks([])
    ax1.set_xticklabels([])

    # Hide y-axis ticks and labels
    ax1.set_yticks([])
    ax1.set_yticklabels([])

#     # Hide the secondary and third y-axes
#     ax2.set_yticks([])
#     ax2.set_yticklabels([])
#     ax3.set_yticks([])
#     ax3.set_yticklabels([])

    # Hide the spines (axes)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_visible(False)
#     ax3.spines['left'].set_visible(False)

    
    output_folder = './imgs/10p5/'
    filename = f'option{name}-{quotedate}.png'
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    
    img = Image.open(filepath).convert('L')

    # Save the grayscale image
    filepath_grayscale = os.path.join(output_folder, f'option{name}-{quotedate}.png')
    img.save(filepath_grayscale)

    
# Example usage
# generate_plot(your_dataframe, 'example_name')

    
# Example usage
# generate_plot(your_dataframe, 'example_name')


# - Now we can generate all images and save them --> MAIN LOOP

# In[47]:

print('Getting ready to loop')


#Main loop -> We partition the dataset for faster boolean indexing
target_return = []
target_labels = []
target_quote_start_date = []
target_description = []

temp_df = df.iloc[0:10000000,:]
unique_options = list(temp_df.unique_id.value_counts().index)
unique_options = sorted(unique_options, key=lambda x: random.random())

for i in tqdm(unique_options[0:10000]):
    indicator=0
    temp = temp_df.loc[df['unique_id'] == i]
    leny = len(temp) // 15
    for idct in range(leny):
        try:
            i_input = temp.iloc[indicator:indicator + 10, :]
            i_target = temp.iloc[indicator+9:indicator+14, :]

            for j in col_float[:-2]:
                i_input[j+'_pct'] = i_input[j].pct_change()

            i_input.fillna(0)

            ret = (i_target['C_BID'].values[-1] - i_target['C_BID'].values[0]) / i_target['C_BID'].values[0]
            target_return.append(ret)
            target_description.append(i)
            target_quote_start_date.append(i_input['QUOTE_DATE'].values[0])

            if i_target['C_BID'].values[0] < i_target['C_BID'].values[-1]:
                target_labels.append(1)
            else:
                target_labels.append(0)

            generate_plot(i_input, i, i_input.QUOTE_DATE.values[0])

            indicator += 15

        except:
            print(f'Couldnt Render {i}')
            break
            
            
target_df = pd.DataFrame({'Label':target_labels, 'Return': target_return, 'Description': target_description, 'Quote_Start_Date': target_quote_start_date})

target_df.to_csv('Target.csv')

print('All Done')