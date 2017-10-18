
# coding: utf-8

# ## Can You Predict Product Backorders?
# 

# ### Project:
# Ultimately, backordering boils down to having orders that you can’t fulfill or more orders than you have stock on hand. It’s a dream for any business but it’s also a huge problem if you don’t know how to handle it.
# In this project we are trying to identify parts at risk of backorder before the event occurs so the business has time to react.

# ### Data:
# Data is Based on historical data predict backorder risk for products
# Training data file contains the historical data for the 8 weeks prior to the week we are trying to predict. 
# The data was taken as weekly snapshots at the start of each week.
# data is from kaggle :
# https://www.kaggle.com/tiredgeek/predict-bo-trial/data

# ##### Data Wrangling and Cleaning:
# The data come in the form of MS Excel spreadsheets, which are easily loaded into pandas dataframes.
# in this data sets we have Yes/No fields which are converted in to binary integers.
# -99 values in performance columns which are missing values and replaced by median.
# we have some NaNs in lead_time but we are not sure that are missing or not.
# It's quite likely that when lead_time is missing, it's missing for a reason and not at random, which means a mean/median imputation strategy may not be appropriate.
# I prefer to decide by looking at data with calculationg the proportion of backordered products with vs. without a missing value in lead_time.
# Since the data set is very big I decided to reduce data by capturing 60% of the total sales volume which is a big reduction in data for not much loss of fidelity.

# ##### Data Visualization
# For data visualization I use transformation(square root) thats appropriate for heavy tailed data.

# In[56]:

import decimal

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.preprocessing import Imputer
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# In[57]:

#train_data_=train_data


# In[58]:

backorder_file = pd.read_csv("Kaggle_Training_Dataset_v2.csv")

orders = (backorder_file
              .drop(backorder_file.index[len(backorder_file)-1])# drop invalid last row
              .replace(['Yes', 'No'], [1, 0]))               # make yes/no numeric


# In[59]:

orders.tail()
#data Information
orders.info()
#data description
orders.describe()


# ### Data Preparation

# In[60]:

#missing values in product
orders.sku.isnull().sum()


# In[61]:

#missing value 
orders.isnull().sum()


# In[62]:

#replacing -99 missing values to median
imp=Imputer(missing_values=-99,strategy='median')
for data in ['perf_6_month_avg','perf_12_month_avg']:
    orders[data] = imp.fit_transform(orders[data].values.reshape(-1, 1))


# ### EDA

# In[88]:

#backorder ratio
prob=len(orders[orders.went_on_backorder==1])/len(orders.sku)
print((prob*100),'%')
print(len(orders[orders.went_on_backorder==1]))


# check the missing data in lead time to replace it or not?
# 1. Proportion of orders that “went_on_backorder” for missing lead_time records
# 2. Proportion of orders that “went_on_backorder” for non-null lead_time records

# In[98]:

n_null_leadTime = orders[orders['lead_time'].isnull()].shape[0]
print ('number of orders with missing lead time:', orders[orders['lead_time'].isnull()].shape[0])
n_non_null_leadTime = orders[orders['lead_time'].notnull()].shape[0]
print ('number of orders without missing lead time:',orders[orders['lead_time'].notnull()].shape[0])
n_null_leadTime_backorders =sum(orders[np.isnan(orders["lead_time"])]["went_on_backorder"])
print ('Number of backordered products with misssing lead time:', n_null_leadTime_backorders)
n_non_null_leadTime_backorders = sum(orders[pd.notnull(orders["lead_time"])]["went_on_backorder"])
print  ('Number of backordered products without misssing lead time:',n_non_null_leadTime_backorders)
print ('Total orders went on backorders:',n_null_leadTime_backorders+ n_non_null_leadTime_backorders)

null_leadTime_backorder_ratio = n_null_leadTime_backorders / float(n_null_leadTime)
non_null_leadTime_backorder_ratio = n_non_null_leadTime_backorders / float(n_non_null_leadTime)
print('Proportion of orders without missing lead time that went_on_backorder:',non_null_leadTime_backorder_ratio * 100)
print('Proportion of orders with missing lead_time that went_on_backorder :', null_leadTime_backorder_ratio * 100)


# Based on the above calculations the proportion of backordered products with missing lead time is 50% less than those without missing lead time.
# The proportion of backordere products with missing lead time is half of the products with no missing values, therefore I decided not to replace the missing data in lead time and not dropiing them.
# Plot below shows the density of products for a given lead time that went on backorder and did not go on backorder.

# In[100]:

#Relationship between lead time and went on backorder
sns.kdeplot(orders[(orders['went_on_backorder'] == 0) & (orders['lead_time'] < 20)]['lead_time'],color='b', shade=True,label='not went on backorder')
sns.kdeplot(orders[(orders['went_on_backorder'] == 1) & (orders['lead_time'] < 20)]['lead_time'], color='r',shade=True,label='went on backorder')
plt.title('Lead Time vs backorder')
plt.show()


# In[145]:

sns.kdeplot(orders[(orders['lead_time'] < 20)]['lead_time'],color='b',shade="True")

plt.show()


# I can see that both lead time with back order and not backorder both peak for the same Lead time,because the orders on that particular lead time is high. However, the backorder graph is lower than no backorder graph.
# You can see that plot of lead time is exactly look like the plot of lead time with not backorder data it means that most of our data is not going on backorder and if we choose random sample of data it will be the same distribution.
# Therefore I am going to see if lead time and went on backorder are independent or dependent from each other.
# Next step is looking at the relationship between lead time and fraction of products that went on backorder to see how lead time changes the probability of went to backorder.

# peak lead time for went on back order, which shows products went bacorder 8 weeks and then 2 weeks lead time have the highest order volumes.

# In[209]:

backorder_df=orders[orders['went_on_backorder']==1]
weekdist=backorder_df['lead_time'].value_counts().sort_index()
weekdist.plot(kind='bar')
_=plt.xlabel('weeks')
_=plt.ylabel('Order volume')
_=plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
plt.title('weekly lead time Volume')
plt.show()


# In[210]:

## These are the peak weeks
weekdist[weekdist>1000]


# In[ ]:




# Plot below shows the relation between lead time and the fraction of backorder. The following plot shows with longer lead time, backorder proportion goes down.

# In[101]:

import decimal
b=orders[['went_on_backorder','lead_time']]
backorder=b[b.went_on_backorder==1]
no_backorder=b[b.went_on_backorder==0]
lead_b=backorder.lead_time.value_counts()
lead_n=no_backorder.lead_time.value_counts()
c=[]
df1 = pd.DataFrame(
    {
     'lead_b': lead_b,
     'lead_n':lead_n
    })

df1=df1[['lead_b','lead_n']].dropna()


for i in range(0,len(df1)):
    backorder_ratio=(df1.lead_b.iloc[i])/(df1.lead_n.iloc[i]+df1.lead_b.iloc[i])
    backorder_ratio=backorder_ratio*100
    c.append(backorder_ratio)


print(np.corrcoef(df1.index.values,c))

plt.plot(df1.index.values,c,"*",color="blue")
plt.title('ratio of lead time(transit time for products)that went on backorder')
#plt.plot(np.array(range(len(c))) * 0.25, c,".")
plt.show()
print(c)
df1


# In the above plot, two outlier are noticed. one is at lead time=11 and one at lead time 52. 
# for the point on 52 I belive there was not enough records to show the rest of point between17 to 52.
# the point at lead time 11 should be given special attention till its cause is known.
# fo this reason I am going to calulate the probability bionomial distribution.

# In[18]:

from scipy.stats import binom
import math
s = binom.pmf(19, 1094,0.01)
print(s)
sd=math.sqrt(1094 * 0.01 * (1 - 0.01))
print("Standard deviation ",sd)
print ((19 - 10.94)/sd)


# As you see from the above calculations standard deviation of bionomial distribution is 3.23 standard deviation from the mean so I am going to ignore this point 

# ### Data Reduction:
# Cumulative percentage is one way of expressing frequency distribution. 
# 

# In[212]:

sales_sort=orders.sort_values('sales_9_month',ascending = False)
sns.kdeplot(np.log(sales_sort['sales_9_month']),color='m', shade=True)
plt.show()


# In[213]:

#relationship between sales and backorder
backorder_sales=sales_sort[sales_sort.went_on_backorder==1]
no_backorder_sales=sales_sort[sales_sort.went_on_backorder==0]
sales_b=backorder_sales.sales_9_month.value_counts()
sales_n=no_backorder_sales.sales_9_month.value_counts()
g=[]
df2 = pd.DataFrame(
    {
     'sales_b': sales_b,
     'sales_n':sales_n
    })

df2=df2[['sales_b','sales_n']].dropna()


for i in range(0,len(df2)):
    backorder_ratio=(df2.sales_b.iloc[i])/(df2.sales_n.iloc[i]+df2.sales_b.iloc[i])
    backorder_ratio=backorder_ratio*100
    g.append(backorder_ratio)


print(np.corrcoef(df2.index.values,g))
plt.plot(np.array(range(len(g))) * 0.25, g,".",color='purple')
plt.title('ratio of total sales count values that went on backorder')
plt.xlabel('sales')
plt.ylabel('ratio of sales went on backorder')
plt.show()


# Data reduction by capture 60% of the total sales volume.

# In[71]:

sales_volume =np.cumsum(sales_sort.sales_9_month)
print(sales_volume[len(sales_volume)-1])
print(0.6 * sales_volume[len(sales_volume)-1])
plt.plot(np.array(range(len(sales_sort))) * 0.28, sales_volume,".",color="orange")
plt.title('Total sales volume')
plt.show()


# In[215]:

volume_perc=0.6 * sales_volume[len(sales_volume)-1]
print(volume_perc)
print(sales_volume[len(sales_volume)-1])
sales_sort['sales_volume']=sales_volume

sales_sort=sales_sort.sort_values('sales_volume',ascending = True)
sample=sales_sort[sales_sort['sales_volume']<=volume_perc]

sample=sample.sort_values('sales_9_month',ascending = False)
plt.plot(np.array(range(len(sample))) * 0.28, sample.sales_9_month,".",color="green")
plt.title('Total sales count')
plt.show()
len(sample)


# So with capturing 60% total sales volume ,data is reduced to 7509 rows.

# How common are backorders? Given that, how likely are backorders based on the part risk flags? 
# And how prevalent are they? 
# What is the relationship between "potential_issue" and "pieces_past_due" are each represented by part 
# risk flags or are they unrelated concepts? What's the relationship between lead time and back orders? 
# Based on the answers to these questions you could recommend: What aspects of the supply chain present the biggest risks? 
# Based on the risks, what would you recommend improving first?
#     
# potential_issue - Source issue for part identified
# 
# pieces_past_due - Parts overdue from source
# 
# local_bo_qty - Amount of stock orders overdue
# 
# deck_risk - Part risk flag
# 
# oe_constraint - Part risk flag
# 
# ppap_risk - Part risk flag

# In[173]:

print(np.corrcoef(orders.potential_issue,orders.pieces_past_due))
print('Source issue for part identified correlation with backorder',np.corrcoef(orders.potential_issue,orders.went_on_backorder))
print('Parts overdue from source correlation with backorder', np.corrcoef(orders.went_on_backorder,orders.pieces_past_due))
print('Amount of stock orders overdue correlation with backorder',np.corrcoef(orders.went_on_backorder,orders.local_bo_qty))
print('oe_constraint - Part risk flag correlation with backorder',np.corrcoef(orders.went_on_backorder,orders.oe_constraint))
print('ppap_risk - Part risk flag correlation with backorder', np.corrcoef(orders.went_on_backorder,orders.ppap_risk))
print('Total sales for perior 9,6,3 and 1 month correlation with backorder',np.corrcoef(orders.went_on_backorder,orders.sales_9_month))
print('Total Forecast sales - Part risk flag correlation with backorder',np.corrcoef(orders.went_on_backorder,orders.forecast_9_month))


# In[144]:

sns.kdeplot(orders[(orders['potential_issue'] == 0) & (orders['lead_time'] < 20)]['lead_time'],color='b', shade=True,label='no potential_issue')
sns.kdeplot(orders[(orders['potential_issue'] == 1) & (orders['lead_time'] < 20)]['lead_time'], color='r',shade=True,label='with potential_issue')
plt.title('Lead Time vs backorder and potential issue from the source')

sns.kdeplot(orders[(orders['went_on_backorder'] == 0) & (orders['lead_time'] < 20)]['lead_time'],color='y', shade=True,label='not went on backorder')
sns.kdeplot(orders[(orders['went_on_backorder'] == 1) & (orders['lead_time'] < 20)]['lead_time'], color='g',shade=True,label='went on backorder')
plt.title('Lead Time vs backorder')

plt.show()


# above plot shows that went on backorder and potential issues have the same relation with lead time. mean when a products with certain lead time did not have the potential isse the products did not go on backorder.
# so potentail issue is one of the main reason that a product went on backorder.

# In[175]:

sns.kdeplot(orders[(orders['ppap_risk'] == 0) & (orders['lead_time']<20)]['lead_time'],color='b', shade=True,label='no Part risk flag')
sns.kdeplot(orders[(orders['ppap_risk'] == 1) & (orders['lead_time']<20)]['lead_time'], color='r',shade=True,label='with Part risk flag')
plt.title('lead time vs ppap_risk')

sns.kdeplot(orders[(orders['went_on_backorder'] == 0) & (orders['lead_time'] < 20)]['lead_time'],color='y', shade=True,label='not went on backorder')
sns.kdeplot(orders[(orders['went_on_backorder'] == 1) & (orders['lead_time'] < 20)]['lead_time'], color='g',shade=True,label='went on backorder')
plt.title('Lead Time vs ppap_risk')

plt.show()


# if the products didnt have part risk flag then it did not go back order.

# In[148]:



sns.kdeplot(orders[(orders['pieces_past_due'] == 0) & (orders['lead_time']<20)]['lead_time'],color='b', shade=True,label='no Parts overdue from source')
sns.kdeplot(orders[(orders['pieces_past_due'] == 1) & (orders['lead_time']<20)]['lead_time'], color='r',shade=True,label=' Parts overdue from source')
plt.title('lead_time vs  Parts overdue from source')

sns.kdeplot(orders[(orders['went_on_backorder'] == 0) & (orders['lead_time'] < 20)]['lead_time'],color='y', shade=True,label='not went on backorder')
sns.kdeplot(orders[(orders['went_on_backorder'] == 1) & (orders['lead_time'] < 20)]['lead_time'], color='g',shade=True,label='went on backorder')
plt.title('Lead Time vs backorder')

plt.show()


# same thing with parts overdue, id there is no parts overdue from the source the products doesnt go backorder

# In[168]:

sns.kdeplot(orders[(orders['went_on_backorder'] == 0) & (orders['forecast_9_month']<10000)]['forecast_9_month'],color='r', shade=True,label='not went on backorder')
sns.kdeplot(orders[(orders['went_on_backorder'] == 1) & (orders['forecast_9_month']<10000)]['forecast_9_month'], color='g',shade=True,label='went on backorder')
plt.title('total forecast vs backorder')
plt.show()


# In[ ]:




# In[172]:

plt.plot((orders.sales_9_month), orders.forecast_9_month,"*",color='darkblue')
plt.title('total sales v.s. total forecast')
plt.xlabel('sales')
plt.ylabel('forecast')
plt.show()
print('Total Forecast sales - Total Sales',np.corrcoef(orders.sales_9_month,orders.forecast_9_month))


# We should find out where actually the forecast was not right so that caused backorder?
# because when saes go up forecast also should go up????
# 

# ## How predictable are sales?
# 

# In[ ]:




# I want to figure out what is the difference of sales and forecast when the product went on backorder.

# In[238]:

fc=orders[orders['went_on_backorder']==1]
#compare sales and forecast
sns.kdeplot(np.log(fc['forecast_9_month']),color='r', shade=True,label='Total forecast')
sns.kdeplot(np.log(fc['sales_9_month']),color='b', shade=False,label='Total sales')
plt.show()


# In[ ]:



