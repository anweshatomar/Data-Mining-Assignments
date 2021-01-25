#!/usr/bin/env python
# coding: utf-8

# In[2]:


#E.1:Work with Pandas module and answer the following questions. Open a .py file and follow the
#instructions and write codes for each section.
#i. Import Pandas and libraries that you think it is needed.
#ii. Import the dataset from BB. The name of the dataset is Data2.txt
import numpy as np
import pandas as pd
chipotle = pd.read_csv("Data2.txt",sep='\t', header=0)
#iii. Assign it to a variable called chipotle and print the 6 observation of it.
print(chipotle.head(6))


# In[3]:


#iv. Clean the item price column and change it in a float data type then reassign the column with
#the cleaned prices.
price = chipotle['item_price']
price=price.str.strip('$')
price=pd.to_numeric(price,downcast='float')
chipotle['item_price']=price
print(chipotle['item_price']) 


# chipotle['item_price'] = chipotle['item_price'].str.replace('$', '')
# chipotle['item_price'] = chipotle['item_price'].astype(float)
# print(chipotle.head(10))

# In[4]:


#v. Remove the duplicates in item name and quantity.
print(chipotle.drop_duplicates(subset=['item_name','quantity']))


# In[5]:


#vi. Find only the products with quantity equals to 1 and find item price that greater that 10.
chipotle1 = chipotle[(chipotle['quantity'] == 1) & (chipotle['item_price'] > 10)]
print(chipotle1)


# In[6]:


#vii. Find the price of each item.
chipotlep = chipotle.loc[(chipotle['quantity'] == 1)]
chipotleprices = chipotlep.groupby(['item_name', 'item_price',]).size().reset_index()
print(chipotleprices)


# In[7]:


#viii. Sort by the name of the item
chipotle.sort_values(by='item_name')


# In[9]:


#ix. find the most expensive item ordered.
ne=chipotle.sort_values('item_price', ascending=True).head(1)
print(ne)


# In[10]:


#x. Find the frequency of times were a Veggie Salad Bowl ordered.
chipotlenamefreq = chipotle.groupby('item_name').get_group('Veggie Salad Bowl')
count=0
for i in chipotlenamefreq:
    count=count+1
print("Frequency of Veggie Salad Bowl",count)


# In[11]:


#xi. How many times people ordered more than one Canned Soda?
chipotle2 = chipotle[(chipotle['quantity']>1)& (chipotle['item_name'] == 'Canned Soda')].count()
print(chipotle2)


# In[12]:


#E.2:Work with Pandas module and answer the following questions. Open a .py file and follow the
#instructions and write codes for each section.
#i. Import Pandas and libraries that you think it is needed.
#ii.Import the dataset from BB. The name of the dataset is Food.txt
import numpy as np
import pandas as pd
food = pd.read_csv("Food.tsv",sep="\t")


# In[13]:


#iii. Print the size of the data frame and the6 observation of it.
size = food.size 
print(size)
print(food.head(6))


# In[14]:


#iv. How many columns this dataset has and print the name of all the columns
column=food.columns
print(column)
c1=food.iloc[0]
print(c1)


# In[15]:


#v. What is the name and data type of 105th column?
c1=food.iloc[105]
print(c1)
print(type(c1))


# In[16]:


#vi.What are the indices of this datset. How they are shaped and ordered.
val=food.index.values
print(val)
print(val.shape)
print(val.size)


# In[17]:


#vii. What is the name of product of 100th observation.
c2=food.iloc[100,7]
print(c2)


# In[18]:


#E.3:Work with Pandas module and answer the following questions. Open a .py file and follow the
#instructions and write codes for each section.
#i. Import Pandas and libraries that you think it is needed.
#ii. Import the dataset from BB. The name of the dataset is Data.txt.
import numpy as np
import pandas as pd
#iii. Assign it to a variable called users and print the 6 observation of it.
users = pd.read_csv("Data.txt",sep="|")
print(users.head(6))


# In[19]:


#iv. Find what is the mean age for occupation.
m=users.sort_values('age').groupby('occupation').mean()
print(m)


# In[28]:


#v. Find the male ratio for occupation and sort it from the most to the least.
m_ratio=users.pivot_table(index='occupation', columns='gender', aggfunc='size')
summ=m_ratio[['F','M']].sum(axis=1)
m_ratio['male_r']=round(m_ratio['M']/summ*100,2)
m_ratio.sort_values('male_r',ascending=False)
print(m_ratio['male_r'])


# In[29]:


#v. Find the male ratio for occupation and sort it from the most to the least.
users1=users[(users.gender == 'M')]
users2 = users1.groupby(['occupation','gender']).mean()
print(users2)
print(type(users2))
users2.sort_values(by='age', ascending=True)


# In[58]:


#vi. For each occupation, calculate the minimum and maximum ages.
print(users.groupby('occupation')['age'].max())
print(users.groupby('occupation')['age'].min())


# In[32]:


#viii. Per occupation present the percentage of women and men
m_ratio['fratio']=round(m_ratio['F']/summ*100,2)
print("*****Female ratio*****")
print(m_ratio['fratio'])
print("*****Male ratio****")
print(m_ratio['fratio'])


# In[59]:


#vii.For each combination of occupation and gender, calculate the mean age.
users2 = users.groupby(['occupation','gender']).mean()
print(users2)


# In[1]:


# =================================================================
# Class_Ex1:
# From the data table above, create an index to return all rows for
# which the phylum name ends in "bacteria" and the value is greater than 1000.
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'value':[632, 1638, 569, 115, 433, 1130, 754, 555],
                     'patient':[1, 1, 1, 1, 2, 2, 2, 2],
                     'phylum':['Firmicutes', 'Proteobacteria', 'Actinobacteria',
    'Bacteroidetes', 'Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes']})
print(data[(data.phylum.str.endswith('bacteria')) & (data.value>1000)])
print('#',50*"-")


# In[2]:


# =================================================================
# Class_Ex2:
# Create a treatment column and add it to DataFrame that has 6 entries
# which the first 4 are zero and the 5 and 6 element are 1 the rest are NAN
# ----------------------------------------------------------------
import pandas as pd
datatreatment = pd.DataFrame({'value':[632, 1638, 569, 115, 433, 1130],
                              'treatment':[0, 0, 0, 0, 1, 1]})

data = pd.merge(data,datatreatment,how='left',on='value')
print(data)
print('#',50*"-")


# In[3]:


# =================================================================
# Class_Ex3:
# Create a month column and add it to DataFrame. Just for month Jan.
# ----------------------------------------------------------------
import sys
import pandas as pd
data['month']='jan'
print(data)
print('#',50*"-")


# In[4]:


# =================================================================
# Class_Ex4:
# Drop the month column.
# ----------------------------------------------------------------
import pandas as pd
data = data.drop(["month"], axis=1)
print(data)
print('#',50*"-")


# In[5]:


# =================================================================
# Class_Ex5:
# Create a numpy array that has all the values of DataFrame.
# ----------------------------------------------------------------
import numpy as np
arr=np.array(data)
print(type(arr))
print(arr)
print('#',50*"-")


# In[6]:


# =================================================================
# Class_Ex6:
# Read baseball data into a DataFrame and check the first and last
# 10 rows
# ----------------------------------------------------------------
bs = pd.read_csv("baseball.csv", header=0)
print(bs.head(10))
print('#',50*"-")


# In[7]:


# =================================================================
# Class_Ex7:
# Create  a unique index by specifying the id column as the index
# Check the new df and verify it is unique
# ----------------------------------------------------------------
bs = bs.set_index('id')
val=pd.Series(bs.index)
print(val.is_unique)
print(val.head(10))
print('#',50*"-")


# In[8]:


# =================================================================
# Class_Ex8:
#Notice that the id index is not sequential. Say we wanted to populate
# the table with every id value.
# Hint: We could specify and index that is a sequence from the first
# to the last id numbers in the database, and Pandas would fill in the
#  missing data with NaN values:
# ----------------------------------------------------------------
indx=bs.index
val=indx.min()
val1=indx.max()
print(val,val1)
newarr = np.arange(val,val1+1,1)
bs = bs.reindex(newarr)
print(bs.head(10))
print('#',50*"-")


# In[9]:


# =================================================================
# Class_Ex9:
# Fill the missing values
# ----------------------------------------------------------------
bs = bs.fillna(method='ffill')
print(bs.head(10))
print('#',50*"-")


# In[10]:


# =================================================================
# Class_Ex10:
# Find the shape of the new df
# ----------------------------------------------------------------
print(bs.shape)
print('#',50*"-")


# In[11]:


# =================================================================
# Class_Ex11:
# Drop row 89525 and 89526
# ----------------------------------------------------------------
bs = bs.drop(89525, axis = 0)
bs = bs.drop(89526, axis = 0)
print(bs.shape)
print('#',50*"-")


# In[12]:


# =================================================================
# Class_Ex12:
# Sort the df ascending and not ascending
# ----------------------------------------------------------------
bs = bs.sort_values('id',ascending=False)
print(bs.head(10))
bs = bs.sort_values('id',ascending=True)
print(bs.head(10))
print('#',50*"-")


# In[ ]:




