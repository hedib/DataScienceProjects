
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

# In[514]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import numpy as np
import matplotlib.mlab as mlab
import scipy.stats as stats
import seaborn as sns
from scipy import stats


# In[515]:


train_data_=train_data


# In[516]:


raw_data = pd.read_csv("Kaggle_Training_Dataset_v2.csv")

train_data = (raw_data
              .drop(raw_data.index[len(raw_data)-1])# drop invalid last row
              .replace(['Yes', 'No'], [1, 0]))   # make yes/no numeric

train_data.tail()


# In[517]:


#data Information
train_data.info()


# In[518]:


#data description
train_data.describe()


# In[519]:


#missing values in product
train_data.sku.isnull().sum()


# In[520]:


#missing value 
train_data.isnull().sum()


# In[524]:


#replacing -99 missing values to median
imp=Imputer(missing_values=-99,strategy='median')
for data in ['perf_6_month_avg','perf_12_month_avg']:
    train_data[data] = imp.fit_transform(train_data[data].values.reshape(-1, 1))


# In[525]:


#backorder ratio
prob=len(train_data[train_data.went_on_backorder==1])/len(train_data.sku)
print((prob*100),'%')


# check the missing data in lead time to replace it or not?
# 1. Proportion of orders that “went_on_backorder” for missing lead_time records
# 2. Proportion of orders that “went_on_backorder” for non-null lead_time records

# In[25]:


n_null_leadTime = train_data[train_data['lead_time'].isnull()].shape[0]
print(n_null_leadTime)
n_non_null_leadTime = train_data[train_data['lead_time'].notnull()].shape[0]
print(n_non_null_leadTime)
n_null_leadTime_backorders =sum(train_data[np.isnan(train_data["lead_time"])]["went_on_backorder"])
print (n_null_leadTime_backorders)
n_non_null_leadTime_backorders = sum(train_data[pd.notnull(train_data["lead_time"])]["went_on_backorder"])
print  (n_non_null_leadTime_backorders)

null_leadTime_backorder_ratio = n_null_leadTime_backorders / float(n_null_leadTime)
non_null_leadTime_backorder_ratio = n_non_null_leadTime_backorders / float(n_non_null_leadTime)
print('Proportion of orders that “went_on_backorder” for no missing lead_time records:',non_null_leadTime_backorder_ratio * 100)
print('Proportion of orders that “went_on_backorder” for missing lead_time records:', null_leadTime_backorder_ratio * 100)


# Based on the proportion of orders for missing lead_time_ time of orders went backorder, we can see that the result is 50% less 
# than proportion of lead time without missing values, and our total went backorder proportion is around 0.66 and it is close to 
# proportion of orders that went back order for non missing values. 
# Therefore i decided to not replacing the missing data 
# below we can see the lead time plot and went_backorder type.

# In[531]:


#coorelation between lead time and wen on backorder
scores = train_data[['lead_time','went_on_backorder']]
sns.countplot(x="lead_time", hue="went_on_backorder", data=scores, palette={1: "r", 0: "g"})
plt.show()


# In[534]:


from sklearn import preprocessing
transfer_lead = preprocessing.scale(np.sqrt(pd.notnull(scores['lead_time'])))
sns.countplot(x=transfer_lead, hue="went_on_backorder", data=scores, palette={1: "r", 0: "g"})
plt.show()


# In[535]:


import decimal
b=train_data[['went_on_backorder','lead_time']]
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


# In[536]:


sales_data=train_data
total_sales=(sales_data.sales_1_month+sales_data.sales_3_month+sales_data.sales_6_month+sales_data.sales_9_month)
sales_data['total_sales']=total_sales
reduced_data=sales_data.sort_values('total_sales',ascending = False)

plt.plot(np.array(range(len(reduced_data))) * 0.28, reduced_data.total_sales,".",color="green")
plt.title('Total sales count')
plt.show()


# In[537]:


#correlation between sales and backorder

sw=reduced_data[['went_on_backorder','total_sales']]
backorder_sales=sw[sw.went_on_backorder==1]
no_backorder_sales=sw[sw.went_on_backorder==0]
sales_b=backorder_sales.total_sales.value_counts()
sales_n=no_backorder_sales.total_sales.value_counts()
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

# In[395]:


sales_volume =np.cumsum(sample_data.total_sales)
print(sales_volume[len(sales_volume)-1])
print(0.6 * sales_volume[len(sales_volume)-1])
plt.plot(np.array(range(len(sample_data))) * 0.28, sales_volume,".",color="orange")
plt.title('Total sales volume')

plt.show()


# In[407]:


a=sample_data
volume_perc=0.6 * sales_volume[len(sales_volume)-1]
print(volume_perc)
print(sales_volume[len(sales_volume)-1])
a['sales_volume']=sales_volume

a=a.sort_values('sales_volume',ascending = True)
a=a[a['sales_volume']<=volume_perc]

a=a.sort_values('total_sales',ascending = False)
plt.plot(np.array(range(len(a))) * 0.28, a.total_sales,".",color="green")
plt.title('Total sales count')
plt.show()
len(a)


# So with capturing 60 % total sales volume ,data is reduced to 7509 rows.

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

# In[419]:


total_forecast=(a.forecast_3_month+a.forecast_6_month+a.forecast_9_month)
a['total_forecast']=total_forecast
print(np.corrcoef(a.potential_issue,a.pieces_past_due))
print('Source issue for part identified correlation with backorder',np.corrcoef(a.potential_issue,a.went_on_backorder))
print('Parts overdue from source correlation with backorder', np.corrcoef(a.went_on_backorder,a.pieces_past_due))
print('Amount of stock orders overdue correlation with backorder',np.corrcoef(a.went_on_backorder,a.local_bo_qty))
print('oe_constraint - Part risk flag correlation with backorder',np.corrcoef(a.went_on_backorder,a.oe_constraint))
print('ppap_risk - Part risk flag correlation with backorder', np.corrcoef(a.went_on_backorder,a.ppap_risk))
print('Total sales for perior 9,6,3 and 1 month correlation with backorder',np.corrcoef(a.went_on_backorder,a.total_sales))
print('Total Forecast sales - Part risk flag correlation with backorder',np.corrcoef(a.went_on_backorder,a.total_forecast))
print('Transit time-Lead time - Part risk flag correlation with backorder', np.corrcoef(a.went_on_backorder,a.lead_time))
print('Total Forecast sales - Total Sales',np.corrcoef(a.total_sales,a.total_forecast))


# In[461]:


b=a.sort_values('total_sales',ascending = True)
plt.plot((b.total_sales), a.total_forecast,"*",color='darkblue')
plt.title('total sales and total forecast')
plt.xlabel('sales')
plt.ylabel('forecast')
plt.show()
#b[['total_sales','total_forecast']].tail(70)


# 
# 
# The next raw uses scale method from scikit-learn to transform the distribution 
# This will not impact Skewness Statistic calculation
# We have included this for sake of completion
# so here I compare three vesion of plots:Original,scale with sqr.

# In[511]:


from scipy.stats import boxcox
from sklearn import preprocessing
from scipy.stats import skew
leadTime = preprocessing.scale(np.sqrt(pd.notnull(a['lead_time'])))

#leadTimeBoxCox = preprocessing.scale(boxcox(pd.notnull(a['lead_time']+5))[0])
leadTimeOrig = preprocessing.scale(pd.notnull(a['lead_time']))
skness = skew(leadTime)
#sknessBoxCox = skew(leadTimeBoxCox)
sknessOrig = skew(leadTimeOrig)


#We draw the histograms 
figure = plt.figure()
figure.add_subplot(131)   
plt.hist(leadTime,facecolor='red',alpha=0.5) 
plt.xlabel("lead_time - Transformed(Using Sqrt)") 
plt.title("Lead_time Histogram") 
plt.text(2,1000,"Skewness: {0:.5f}".format(skness)) 


figure.add_subplot(133) 
plt.hist(AirTimeOrig,facecolor='green',alpha=0.5) 
plt.xlabel("lead_time - Based on Original lead Times") 
plt.title("lead_time Histogram - Right Skewed") 
plt.text(2,1000,"Skewness: {0:.5f}".format(sknessOrig))
plt.show()

