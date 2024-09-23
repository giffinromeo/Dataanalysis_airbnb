#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import warnings
import matplotlib
from scipy.stats import skew
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from scipy import stats
import statsmodels.stats.multitest as multi

df = pd.read_csv('./listings.csv')


# In[37]:


df1 = pd.read_csv('./calendar.csv')


# In[38]:


df2 = pd.read_csv('./reviews.csv')


# In[39]:


df.head()


# In[ ]:


####Data Preparation


# In[40]:


df_listing_last_scraped = pd.Timestamp(df.last_scraped[0])


# In[41]:


print(df_listing_last_scraped)


# In[42]:


most_missing_cols=set(df.columns[df.isnull().mean() > 0.90])#Provide a set of columns with more than 75% of the values missing


# In[43]:


df = df.drop(most_missing_cols, axis=1)


# In[44]:


#df1.isnull().sum()


# In[45]:


df1.isnull().mean()


# In[46]:


df1['price'] = pd.to_numeric(df1['price'].replace('[\$,]', '', regex=True), errors='coerce')
df1.dropna(subset=['price'], inplace=True)
print(df1)


# In[47]:


#converting and relpacing the dollar symbol and comma from the cost incurred 
df.extra_people = df.extra_people.str.replace(r"$","").str.replace(",","").astype("float32")
df.price = df.price.str.replace(r"$","").str.replace(",","").astype("float32")
df.cleaning_fee = df.cleaning_fee.str.replace(r"$","").str.replace(",","").astype("float32")
df.weekly_price = df.weekly_price.str.replace(r"$","").str.replace(",","").astype("float32")
df.security_deposit = df.security_deposit.str.replace(r"$","").str.replace(",","").astype("float32")
df.monthly_price = df.monthly_price.str.replace(r"$","").str.replace(",","").astype("float32")


# In[48]:


#droping out columns containing same value throughout doesnt brin much on to the table
df.drop([c for c in df.columns if df[c].nunique()==1],axis=1,inplace=True)


# In[49]:


print(df.shape[1])


# In[50]:


df.columns.values


# In[51]:


#extracting usefull date time from the given dates
df['host_since_time difference'] = (pd.to_datetime(df_listing_last_scraped)-pd.to_datetime(df.host_since)).dt.days
df['last_review_time difference'] = (pd.to_datetime(df_listing_last_scraped)-pd.to_datetime(df.last_review)).dt.days
df['first_review_time difference'] = (pd.to_datetime(df_listing_last_scraped)-pd.to_datetime(df.first_review)).dt.days


# In[52]:


# Reformat other features
#df_listings['host_response_time'] = df_listings.host_response_time.map({"within an hour":1,"within a few hours":12,\
                                                                       # "within a day":24,"a few days or more":48})
df.host_response_rate = df.host_response_rate.str.replace("%","").astype("float32")
##df_listings['cancellation_policy'] = df_listings['cancellation_policy'].map({'strict':0,'moderate':1,'flexible':2})

# Create a feature count the number of host verification methods
host_verifications = np.unique(np.concatenate(df.host_verifications.map(lambda x:x[1:-1].replace("'","").split(", "))))[1:]
matrix_verifications=[[ite in row for row in df.host_verifications.map(lambda x:x[1:-1].replace("'","").split(", ")) ] for ite in host_verifications]
df['host_verification_possibility_count'] = pd.DataFrame(matrix_verifications,index=host_verifications).T.sum(1)

#.T:This transposes the DataFrame, swapping rows and columns. Now, each row corresponds to a host, and each column corresponds to a verification method.
#.sum(1):This sums the values along each row (i.e., for each host). Since True is treated as 1 and False as 0, this effectively counts the number of True values (verification methods) for each host.


# In[53]:


print(df['host_verification_possibility_count'])


# In[54]:


features_host = ['host_is_superhost','host_about','host_response_time','host_response_rate', 'host_listings_count', 'host_verification_possibility_count','host_has_profile_pic','host_identity_verified','host_since_time difference', 'calculated_host_listings_count']


# In[55]:


features_property = ['summary','space','description','neighborhood_overview','notes','transit',
                     'street','neighbourhood','zipcode','latitude','longitude','is_location_exact',
                     'property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type',
                     'amenities','price','weekly_price','security_deposit','cleaning_fee',
                     'guests_included','extra_people','minimum_nights','maximum_nights']


# In[56]:


features_traveler = ['number_of_reviews','last_review_time difference','first_review_time difference','review_scores_rating', 'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication', 'review_scores_location','review_scores_value','instant_bookable','cancellation_policy', 'require_guest_profile_picture','require_guest_phone_verification','reviews_per_month']


# In[57]:


features = features_host + features_property + features_traveler
df_final=df[features]


# In[58]:


numeric_feature = ['host_listings_count','calculated_host_listings_count','latitude','longitude','accommodates','bathrooms',
                   'bedrooms','beds','guests_included','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating',\
                   'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication',
                   'review_scores_location','review_scores_value', 'review_scores_rating' ,'review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location',
                   'review_scores_value' ,'reviews_per_month'] 


# In[59]:


bool_feature = ['host_is_superhost','host_has_profile_pic','host_identity_verified','is_location_exact','instant_bookable',
                'require_guest_phone_verification','require_guest_profile_picture']


# In[60]:


#converting it into boolean feature
for bool_f in bool_feature:
    df_final[bool_f] = df_final[bool_f].map({'t':1,'f':0}).astype('bool')

# Transform the numerical features
for num_f in numeric_feature:
    df_final[num_f] = df_final[num_f].astype("float32")


# In[61]:


# Fix the weird zipcode value
df_final.zipcode[df_final.zipcode=="99\n98122"] = 98122


# In[62]:


# Transform the amenities feature into a one-hot encoding matrix
unqiue_amenities = np.unique(np.concatenate(df_final.amenities.str[1:-1].str.replace('"','')                                            .str.split(",")))[1:]
matrix_amenities=[[amen in row for row in df_final.amenities.str[1:-1].str.replace('"','').                   str.split(",") ] for amen in unqiue_amenities]
df_amenities = pd.DataFrame(matrix_amenities,index=unqiue_amenities).T

# Drop amenities features appaer in less than 5% of samples to avoid overfitting
df_amenities.drop(df_amenities.columns.values[np.where(df_amenities.mean()<0.05)],axis=1,inplace=True)
df_listings_filtered = pd.concat([df_final,df_amenities],axis=1)


# In[63]:


df_listings_filtered = df_listings_filtered.query('number_of_reviews>0')
print(df_listings_filtered.isna().mean().sort_values(ascending=False).head(35))


# In[64]:


df_listings_filtered_cleaned = df_listings_filtered.copy()

# Fill NA for numeric features
df_listings_filtered_cleaned.zipcode = df_listings_filtered_cleaned.zipcode.fillna(
                                                df_listings_filtered_cleaned.zipcode.mode()[0])
feature_fillna_median = ['host_response_time','host_response_rate','security_deposit','cleaning_fee','weekly_price','bedrooms',
                         'bathrooms','review_scores_rating','review_scores_communication','review_scores_cleanliness','review_scores_location',
                         'review_scores_value','review_scores_accuracy','review_scores_checkin']
df_listings_filtered_cleaned[feature_fillna_median] = df_listings_filtered_cleaned[feature_fillna_median].fillna(
                                                                df_listings_filtered_cleaned[feature_fillna_median].median())

# Fill NA for object features
feature_fillna_empty =  ['summary','neighbourhood','space','host_about','transit','neighborhood_overview','notes']
df_listings_filtered_cleaned[feature_fillna_empty] = df_listings_filtered_cleaned[feature_fillna_empty].fillna('')

# Numerical features
df_num = df_listings_filtered_cleaned.select_dtypes(exclude='object')
# One hot encoding categorical features
df_cat = pd.get_dummies(df_listings_filtered_cleaned.select_dtypes(include='object')[['property_type','room_type', 'bed_type']])
# Drop one hot categorical feature columns appearing less then 5% of samples
catFeatureToDrop = df_cat.columns.where(df_cat.mean()<0.05).dropna()
df_cat.drop(catFeatureToDrop,axis=1,inplace=True)
df_cat = df_cat.astype("bool")
df_total = pd.concat([df_num,df_cat],axis=1)


# In[65]:


#### Modeling


# In[66]:


# Create the metric and named it "performance"
df_total['performance'] = df_total.reviews_per_month * df_num.review_scores_rating
# Draw host and review related features
featureToDrop = [f for f in df_total.columns.values if "review" in f or 'host' in f]
featureToDrop


# In[67]:


X = df_total.drop(featureToDrop+['performance'],axis=1)
y = df_total.performance
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 10,100],scoring='r2').fit(X, y)
print("r2 score:{:.3f}".format(clf.score(X,y)))


# In[68]:


matplotlib.rcParams['figure.figsize'] = (6,6)
fig = plt.figure()

performance = df_total['performance']
log_performance = np.log1p(performance)
# histogram of performance 
ax = fig.add_subplot(2, 2, 1)
performance.hist(ax=ax)
plt.title("performance")

# qq-plot of performance
ax = fig.add_subplot(2, 2, 2)
sm.qqplot((performance-performance.mean())/performance.std(),line='45',ax=ax)
plt.title("performance")

# histogram of log transformed performance
ax = fig.add_subplot(2, 2, 3)
log_performance.hist(ax=ax)
plt.title("log(1+performance)")

# qqplot of log transformed performance
ax = fig.add_subplot(2, 2, 4)
sm.qqplot((log_performance-log_performance.mean())/log_performance.std(),line='45',ax=ax)
plt.title("log(1+performance)")
plt.tight_layout()


# In[69]:


# Choose numeric features
num_feature = df_total.select_dtypes(include="number").dtypes.index.values

# Compute the sknewness, log transform features with abs(skewness)>0.75
skewed_feats = df_total[num_feature].apply(lambda x:x.skew())
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
skewed_feats = skewed_feats.index
# Helper function transforming features containing negative values 
# to features only containing nonnegative values
def moveToNonNegative(series):
  if series.min()<0:
    series = series-series.min()
  return series 

df_total[skewed_feats] = df_total[skewed_feats].apply(moveToNonNegative)
df_total[skewed_feats] = np.log1p(df_total[skewed_feats])


# In[70]:


X = df_total.drop(featureToDrop+['performance'],axis=1)
y = df_total.performance
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 10,100],scoring='r2').fit(X, y)
print("r2 score:{:.3f}".format(clf.score(X,y)))


# In[71]:


coef = pd.Series(clf.coef_, index = X.columns)
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()


# In[72]:


#### Airbnb development in Seattle


# In[73]:


# Extract year and month from date feature
df2.date= pd.to_datetime(df2.date)
df2['year'] = df2.date.dt.year
df2['month'] = df2.date.dt.month
df2.head()


# In[74]:


review_count = df2.groupby(['year','month'])['comments'].size().reset_index()
# We drop 2009 and 2016 here as they don't contains data for full 12 months
review_count = review_count.loc[(review_count.year<2016) & (review_count.year>2009),:]
# I use log transform here to better observe the seasonal trend between different years
review_count_log = review_count.copy()
review_count_log.comments = np.log(review_count_log.comments)


# In[75]:


sns.pointplot(data=review_count_log,x='month',hue='year',y='comments',marker='o',kind='line',palette="RdPu_r",aspect=10/8)
ax = plt.gca()
ax.set_ylabel("log(review_count)")
ax.set_title("Log-transformed Monthly Reviews from 2010 to 2015")
plt.show()


# In[76]:


#### Does it matter to br hosted by a superhost


# In[77]:


# Helper function for two independent sample t-test
def ttest(df,group_feature, test_feature):
    flag0 = df[group_feature]==False
    flag1 = ~flag0
    vector_0 = df.loc[flag0,test_feature]
    vector_1 = df.loc[flag1,test_feature]
    statistic, pvalue = stats.ttest_ind(vector_1, vector_0)
    return [statistic,pvalue,test_feature]    


# In[78]:


# T-test
ttest_result = []
for col in df_total.columns:
    if col=="host_is_superhost":
        continue
    else:
        ttest_result.append(ttest(df_total,"host_is_superhost",col))

# Display the t-test result
ttest_result = pd.DataFrame(ttest_result,columns=['statistics','pvalue','feature'])
# P-value adjustment
multitest_result = multi.multipletests(ttest_result.pvalue,method="bonferroni")
ttest_result['significant'],ttest_result['adjust_pvalue']=multitest_result[0],multitest_result[1]
ttest_result.sort_values(['significant','adjust_pvalue'],ascending=[False,True]).style.bar(subset=['statistics'], 
                                                                     align='zero', color=['#d65f5f', '#5fba7d'])


# In[79]:


# About 43% of the significant features are amenities.
ttest_result.feature[ttest_result.significant==True].isin(unqiue_amenities).sum()/np.sum(ttest_result.significant==True)


# In[80]:


matplotlib.rcParams['figure.figsize'] = (6,6)
plt.subplot(221)
sns.barplot(data=df_total,y="Shampoo",x="host_is_superhost")
plt.title("Shampoo")
plt.subplot(222)
sns.barplot(data=df_total,y="number_of_reviews",x="host_is_superhost")
plt.title("number_of_reviews")
plt.subplot(223)
sns.barplot(data=df_total,y="calculated_host_listings_count",x="host_is_superhost")
plt.title("calculated_host_listings_count")
plt.subplot(224)
sns.barplot(data=df_total,y="price",x="host_is_superhost")
plt.title("price")
plt.tight_layout()

