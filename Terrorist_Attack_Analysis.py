#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv(r'C:\Users\Muskan Khan\OneDrive\Documents\JUPYTER\New folder\globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
print(data)


# In[3]:


data.head()


# In[4]:


# data.info()
data.describe()


# In[ ]:





# In[5]:


null_col=data.isnull().all()
print(null_col)


# In[6]:


# Finding completly null columns:
completly_null_col=null_col[null_col].index.tolist()
print(completly_null_col)


# In[7]:


# Finding percentage of columns which are above 50

null_percent=data.isnull().mean()*100
high_null_col=null_percent[null_percent>50] 

print("Columns with more than 50% null values:")
print(high_null_col)


# In[8]:


# Dropping those, above 50

high_null_col=null_percent[null_percent>50].index.tolist() # for dropping purpose
data_cleaned=data.drop(columns=high_null_col)


data_cleaned.head()


# In[9]:


print(f"Original shape: {data.shape}")
print(f"Cleaned shape: {data_cleaned.shape}")


# In[10]:



null_percent=data_cleaned.isnull().mean()*100
high_null_col=null_percent[null_percent>50] 
print("Columns with more than 50% null values:")
print(high_null_col)


# In[11]:


data_cleaned.head()


# In[12]:


import plotly_express as px


# In[13]:


yearly_counts=data_cleaned['iyear'].value_counts().sort_index() #and this is the year data(whole)

# yearly_counts.index would return [1970, 1971, 1972, 1973], which will be plotted on the x-axis.
# and .values would return [10, 15, 5, 20], which will be plotted on the y-axis.

fig=px.line(yearly_counts,x=yearly_counts.index,y=yearly_counts.values, title='Number of Terrorism Incidents per Year',labels={'y': "Number of Attacks"})
fig.show()


# In[14]:


# Types of attackes(frequency)

attack_types=data_cleaned['attacktype1_txt'].value_counts()
fig=px.bar(attack_types,x=attack_types.index,y=attack_types.values,
           title='Attacks Type Frequency', labels={'x':'Attack Types',
                                                  'y':'Attack Type Frequency'})
fig.show()


# In[15]:


fig=px.line(attack_types,x=attack_types.index,y=attack_types.values,
           title='Attacks Type Frequency', labels={'x':'Attack Types',
                                                  'y':'Attack Type Frequency'})
fig.show()


# In[16]:


target_types=data_cleaned['targsubtype1_txt'].value_counts()

fig=px.line(target_types,x=target_types.index,y=target_types.values, title='Target Type Frequency', labels={   'y':'Target Type Frequency'})
                                           
fig.show()


# In[17]:


country=data_cleaned['country_txt'].value_counts()

fig=px.line(country, x=country.index,y=country.values,title='Target Countries', labels={   'y':'Target Countries'})
fig.show()


# In[18]:


country_counts=data_cleaned.groupby(['country_txt', 'latitude', 'longitude']).size().reset_index(name='Incident Count')


fig=px.density_mapbox(country_counts,
                      lat='latitude',
                      lon='longitude',
                      z='Incident Count',
                      hover_name='country_txt',
                      title='Terrorism Incidents by Country',
                      color_continuous_scale='Viridis',
                      radius=10)
                      
                      
fig.update_layout(mapbox_style='open-street-map',
                 mapbox_zoom=1,
                 mapbox_center={'lat':20,'lon':0})
fig.show()


# In[19]:


# heatmap of countries
# succeed out of whole


# In[20]:


# adding cities in countries


country_counts=data_cleaned.groupby(['country_txt','city', 'latitude', 'longitude']).size().reset_index(name='Incident Count')


fig=px.density_mapbox(country_counts,
                      lat='latitude',
                      lon='longitude',
                      z='Incident Count',
                      hover_name='city',
                      hover_data={'country_txt':True},
                      title='Terrorism Incidents by Country',
                      color_continuous_scale='Viridis',
                      radius=10)
                      
                      
fig.update_layout(mapbox_style='open-street-map',
                 mapbox_zoom=1,
                 mapbox_center={'lat':20,'lon':0})
fig.show()


# In[21]:


success_ratio=data_cleaned['success'].value_counts().sort_index()

success_ratio_df=success_ratio.reset_index() #When you call reset_index() on this Series, it converts the Series into a DataFrame.
success_ratio_df.columns=['index','Success Attack']
success_ratio_df['index']=success_ratio_df['index'].map({1: 'successful',0: 'Unsuccessful'})


fig=px.bar(success_ratio_df, x='index',y='Success Attack',title='Success Attacks Ratio ',labels={'index': 'Attack Type', 'Success Attack': 'Count'})
fig.update_traces(hovertemplate='Attack Type: %{x}<br>Count: %{y}')
fig.show()


# # Classification

# Gonna predict  attack types or likelihood of attacks.

# In[22]:


# cz I wanted a list of all my columns
column_list = data_cleaned.columns.tolist()
print(column_list)


# In[23]:


for column in data.columns:
    print(column)


# In[24]:


data_cleaned.shape


# # Encoding 'country_txt'

# In[25]:


# from sklearn.preprocessing import OneHotEncoder

# # dropping na values(if present)
# data_cleaned=data_cleaned.dropna(subset=['country_txt'])

# encoder=OneHotEncoder()
# country_encoded=encoder.fit_transform(data_cleaned[['country_txt']]).toarray()
# country_df=pd.DataFrame(country_encoded,columns=encoder.get_feature_name_out(['country_txt']))

# data_cleaned=pd.concat([data_cleaned, country_df], axis=1)


# In[26]:


from sklearn.model_selection import train_test_split
x=data_cleaned[['iyear',"imonth","country",'region','nkill', 'nwoundus']]
y=data_cleaned['attacktype1']
x_train,x_test, y_train,y_test=train_test_split(x,y, test_size=0.10, random_state=42)


# # Creating Pipeline

# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

x_train_new=pipeline.fit_transform(x_train)


# # Choosing a Model

# In[28]:


# !pip install scikit-learn xgboost lightgbm catboost tensorflow


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
# from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb



# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
# model= GradientBoostingRegressor()
# model=SVR()
# model=xgb.XGBRegressor()
# model=cb.CatBoostRegressor(verbose=0)

model.fit(x_train_new,y_train)


# In[30]:


some_xt_data=x_test[:5]
some_yt_data=y_test[:5]
prepared_data=pipeline.transform(some_xt_data)


# In[31]:


# nan_exists=np.isnan(some_xt_data).any
# print(nan_exists)


# In[32]:


predictions=model.predict(prepared_data)
print(predictions)
list(some_yt_data)


# In[33]:


comparison=pd.DataFrame({'Actual':some_yt_data,'Predicted':predictions})
print(comparison)


# # Measuring the model's Accuracy
# 

# In[34]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
my_pred_model=model.predict(x_train_new)
mae=mean_absolute_error(y_train,my_pred_model)
mse=mean_squared_error(y_train, my_pred_model)

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)


# In[35]:


rmse=np.sqrt(mse)
print(rmse)


# In[36]:


#the model is not predicting good enough!The is good for analysis!


# # Applying the model on test data

# In[37]:


x_test_prep=pipeline.transform(x_test)
test_pred=model.predict(x_test_prep)

comparison=pd.DataFrame({'Actual':y_test,'Predicted':test_pred})
print(comparison)


# In[38]:


t_msa=mean_absolute_error(y_test,test_pred)
t_mse=mean_squared_error(y_test,test_pred)
t_rmse=np.sqrt(t_mse)


print(t_msa)
print(t_mse)
print(t_rmse)


# # Cross Validation

# In[39]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x_train_new,y_train)
rmse_score=np.sqrt(score)

print(rmse_score)


# In[40]:


y_train.min()


# In[41]:


y_train.max()


# In[42]:


from joblib import load, dump
dump(model, 'Terrorist_Attack_Analysis')


# In[55]:


#Testing of joblib saved model
model=load('Terrorist_Attack_Analysis')
feat=np.array([[2007, 7, 98,7,56,67 ]])
model.predict(feat)


# # Creating mapping dictionary for the numerical prediction to text
# 

# In[1]:


# Creating the attack type mapping dictionary programmatically

# attack_type_mapping=dict(zip(data_cleaned['attacktype1'].unique(),data_cleaned['attacktype1_txt'].unique()))
# model=load('Terrorist_attack_Analysis')


# def predict_attack_type(feat):

#     prediction=model.predict(feat)[0]
#     prediction_attack_type=attack_type_mapping.get(prediction,'Unknown')
#     return prediction_attack_type

# # feat=np.array([[2007, 7, 98,7,56,67 ]])
# feat=np.array([[1970, 6, 102,10,0,0 ]])

# predicted_attack_type=predict_attack_type(feat)
# print(f"Predicted Attack Type: {predicted_attack_type}")


# In[ ]:





# In[ ]:




