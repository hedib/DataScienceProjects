
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

"""
As a best practice always put imports at the top of the file.
It often aids readability to organize imports by type and project.
For example, core python vs. established packages vs. utilities.
"""


import decimal

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.preprocessing import Imputer
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# # Data

# Good to separate config / constants out
backorder_file = 'Kaggle_Training_Dataset_v2.csv'
orders = pd.read_csv(backorder_file, skipfooter=1)

"""
"orders" or some other name is preferable to "orders" because if you train
a supervised learning model, you'll likely split up the data set into at least
a training set and a test set, which makes the "orders" name confusing.
"""

orders.tail()
orders.info()
orders.describe()

# # Data Preparation

orders = orders.replace(['Yes', 'No'], [1, 0])

# In[520]:

#missing value 
orders.isnull().sum()

# In[524]:

#replacing -99 missing values to median
imp=Imputer(missing_values=-99,strategy='median')
for data in ['perf_6_month_avg','perf_12_month_avg']:
    orders[data] = imp.fit_transform(orders[data].values.reshape(-1, 1))


# # EDA

# In[525]:

#backorder ratio
prob=len(orders[orders.went_on_backorder==1])/len(orders.sku)
print((prob*100),'%')


# check the missing data in lead time to replace it or not?
# 1. Proportion of orders that “went_on_backorder” for missing lead_time records
# 2. Proportion of orders that “went_on_backorder” for non-null lead_time records

# In[25]:


n_null_leadTime = orders[orders['lead_time'].isnull()].shape[0]
print(n_null_leadTime)
n_non_null_leadTime = orders[orders['lead_time'].notnull()].shape[0]
print(n_non_null_leadTime)
n_null_leadTime_backorders =sum(orders[np.isnan(orders["lead_time"])]["went_on_backorder"])
print (n_null_leadTime_backorders)
n_non_null_leadTime_backorders = sum(orders[pd.notnull(orders["lead_time"])]["went_on_backorder"])
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
""" 
"Relationship" would be a better word here than correlation
Although in plain language correlation makes sense, when doing
analysis "correlation" refers to a specific statistical measure
and it's good to stick to the correct usage to not confuse others.

And the definition of "scores" isn't necessary.
"""
sns.countplot(x="lead_time", hue="went_on_backorder", data=orders, palette={1: "r", 0: "g"})
plt.show()

"""
Look at the difference between the plot above and the code below
creates. I'd argue the one below is much more readable because
it allows me to compare distribution in lead_time by whether or
not a product went_on_backorder. For the plot above, my eyes
simply aren't good enough to read the plot because the scales (both
on the x-axis and y-axis) are so different. There are, of course,
downsides to KDE plots as opposed to histograms, but in most cases
when comparing distributions KDE's are much easier to read than
histograms. Histograms tend to be preferable only when it's important
to be able to read the actual count off a plot.

Are there any points that jump out to you based on the plot below?
I saw one that piqued my interest ...
"""
sns.kdeplot(orders[(orders['went_on_backorder'] == 0) & (orders['lead_time'] < 20)]['lead_time'], shade=True
sns.kdeplot(orders[(orders['went_on_backorder'] == 1) & (orders['lead_time'] < 20)]['lead_time'], shade=True)


# In[534]:

# I'm not sure what you're trying to show here ...
transfer_lead = preprocessing.scale(np.sqrt(pd.notnull(scores['lead_time'])))
sns.countplot(x=transfer_lead, hue="went_on_backorder", data=scores, palette={1: "r", 0: "g"})
plt.show()


# In[535]:

"""
I recommend separating these 2 steps out and adding commentary.
I know you know the answers to the following questions, but ...
- Why look at this in the first place?
- What does the relationship imply?
- Are there funny points? What do they mean? How should they be handled?
- Based on your conclusion, might you improve your analysis of the relationship?
"""
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


# In[536]:


sales_data=orders
total_sales=(sales_data.sales_1_month+sales_data.sales_3_month+sales_data.sales_6_month+sales_data.sales_9_month)
sales_data['total_sales']=total_sales
reduced_data=sales_data.sort_values('total_sales',ascending = False)

plt.plot(np.array(range(len(reduced_data))) * 0.28, reduced_data.total_sales,".",color="green")
plt.title('Total sales count')
plt.show()


"""
Don't be shy about manipulating a data frame. It can become expensive to make
many copies of a data frame. Pandas will try to use references and avoid
unnecessary copying, but even in the situations where pandas gets that right
for your sake, it's *usually* easier to modify one dataframe than lots of
slightly difference dataframes.

See below for another representation that I'd argue is more readable than
the plot above. Does the log-scale transformation make sense? 

Also, when I read the data dictionary, it looks to me like the sales_9_month
is *cumulative* i.e. all of the last 9 monts of sales, so I don't think the
'total_sales' column is necessary.
"""
train_data['total_sales'] = train_data.sales_1_month + train_data.sales_3_month + train_data.sales_6_month + train_data.sales_9_month
sns.kdeplot(np.log(train_data['total_sales']), shade=True)




# In[537]:


"""
This could probably either be condensed or could use some editorializing.
If all you want to demonstrate is the data reduction aspect, you can
do that with 1 plot of # of products vs. total sales, rather than 3.
If you're getting at something further with these plots, some Q&A would
help me (or the reader) follow along.
"""

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



"""
I'm not quite sure what you're going after here, but I'm going to guess
and give you my thoughts. 

When you want to sqrt transform a distribution
it isn't necessary to fit a scaler, unless your intention is to wrap
it up all in a sklearn pipeline.

In general, transformations (Box-Cox or otherwise) are best used only
if you need the transformation. For plots, a Box-Cox transformation
makes the plot *very* difficult for someone else to read. Where, as
an alternative, most statisticians and scientists will happily read
a log plot. For models, "normalizing" transforms like Box-Cox will be
important for some models (like linear models) but not for others like
tree-based models.

"""

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

