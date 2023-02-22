import pandas as pd
import streamlit as st 
import xgboost as xgb
import numpy as np
import joblib

df_center = pd.read_csv('./fulfilment_center_info.csv').sort_values(by='center_id')
df_meal = pd.read_csv('./meal_info.csv').sort_values(by='meal_id')
df_train = pd.read_csv('./train.csv').drop(['id','checkout_price'],axis=1)

X_test = pd.read_csv('./processed_test_data.csv',index_col=0)

dummy_cols_center_id = [f'center_id_{i}' for i in df_center['center_id']]
dummy_cols_city_code = [f'city_code_{i}' for i in df_center['city_code']]
dummy_cols_region_code = [f'region_code_{i}' for i in df_center['region_code']]
dummy_cols_center_type = [f'center_type_{i}' for i in df_center['center_type']]
dummy_cols_meal_id = [f'meal_id_{i}' for i in df_meal['meal_id']]
dummy_cols_category = [f'category_{i}' for i in df_meal['category']]
dummy_cols_cuisine = [f'cuisine_{i}' for i in df_meal['cuisine']]
all_dummy_cols = dummy_cols_center_id + dummy_cols_city_code + dummy_cols_region_code + dummy_cols_center_type + dummy_cols_meal_id + dummy_cols_category + dummy_cols_cuisine

num_cols = ['op_area','base_price','num_orders_shift1','num_orders_shift2','num_orders_shift3','num_orders_shift4','num_orders_shift5','num_orders_shift6']
scalers = {col: joblib.load(f'./scalers/{col}.scaler') for col in num_cols}

centers = list(df_center['center_id'])
meals = list(df_meal['meal_id'])

centers_pretty = list(df_center['center_id'].astype(str) + ' (City: ' + df_center['city_code'].astype(str) + ', Region: ' + df_center['region_code'].astype(str) + ', Type: ' + df_center['center_type'] + ')') 
meals_pretty = list(df_meal['meal_id'].astype(str) + ' (Category: ' + df_meal['category'] + ', Cuisine: ' + df_meal['cuisine'] + ')') 

xgbmodel = xgb.XGBRegressor()
xgbmodel.load_model('./forecastmodel.json')
st.title('Forecasting Multi-Product/Multi-Location Food Demand')
st.header('Center and Meal for which to forecast demand')

center_col,meal_col = st.columns(2)

selected_center = -1
selected_meal = -1

with center_col:
	selected_pretty_center = st.selectbox('Center: ',centers_pretty)
	selected_center = int(selected_pretty_center.split(' ')[0])
	
with meal_col:
	selected_pretty_meal = st.selectbox('Meal: ',meals_pretty)
	selected_meal = int(selected_pretty_meal.split(' ')[0])
	
df_train_sub = df_train.query('center_id==@selected_center and meal_id==@selected_meal')
st.dataframe(df_train_sub.drop(['center_id','meal_id'],axis=1).reset_index(drop=True),use_container_width=True)

st.header('Forecasting demand')
st.subheader('(Last known week\'s demand: 145)')
total_previous_weeks = 0
previous_weeks = []
total_forecast_weeks = 0
forecast_weeks = []
plot_options_col1,plot_options_col2 = st.columns(2)

emailer = 0
featured = 0

base_price = 0

with plot_options_col1:
	total_previous_weeks = st.slider('Total previous weeks to plot:',0,145,value=10)
	previous_weeks = list(range(145-total_previous_weeks,146))
	price_lower_bound = float(round((df_train_sub['base_price'].min()-0.1*df_train_sub['base_price'].min())))
	price_upper_bound = float(round((df_train_sub['base_price'].max()+0.1*df_train_sub['base_price'].max())))
	base_price = st.slider('Base price:',price_lower_bound,price_upper_bound,value=float(df_train_sub['base_price'].values[-1]))
	
with plot_options_col2:
	total_forecast_weeks = st.slider('Total weeks ahead to forecast:',1,25,value=5)
	forecast_weeks = list(range(146,146+total_forecast_weeks))
	featured = int(st.checkbox('Homepage feature'))
	emailer = int(st.checkbox('Emailer for promotion'))
	
all_weeks = previous_weeks + forecast_weeks
previous_num_orders = list(df_train_sub[df_train_sub['week'].isin(previous_weeks)]['num_orders']) + [None for i in forecast_weeks]
forecast_num_orders = [0 for i in forecast_weeks]

# Process first input for forecasting
x = X_test.loc[df_train_sub.index[-1]].to_frame().transpose().copy().reset_index(drop=True)
x['week'] = 145.0/52.0
for i in range(6):
	week_val = 145-i
	week_df = df_train_sub.query('week==@week_val')
	if len(week_df) == 0:
		x.loc[0,f'num_orders_shift{i+1}'] = scalers[f'num_orders_shift{i+1}'].transform([[0]])[0][0]
	else:
		x.loc[0,f'num_orders_shift{i+1}'] = scalers[f'num_orders_shift{i+1}'].transform([[week_df['num_orders'].values[0]]])[0][0]
x['base_price'] = scalers['base_price'].transform([[base_price]])
x['emailer_for_promotion'] = emailer 
x['homepage_featured'] = featured 
print('printing x',flush=True)
print(x[['week','num_orders_shift1','num_orders_shift2']],flush=True)
forecast_num_orders[0] = xgbmodel.predict(x)[0]
for i in range(1,total_forecast_weeks):
	x.loc[0,'num_orders_shift6'] = x.loc[0,'num_orders_shift5']
	x.loc[0,'num_orders_shift5'] = x.loc[0,'num_orders_shift4']
	x.loc[0,'num_orders_shift4'] = x.loc[0,'num_orders_shift3']
	x.loc[0,'num_orders_shift3'] = x.loc[0,'num_orders_shift2']
	x.loc[0,'num_orders_shift2'] = x.loc[0,'num_orders_shift1']
	x.loc[0,'num_orders_shift1'] = scalers['num_orders_shift1'].transform([[forecast_num_orders[i-1]]])[0][0]
	x.loc[0,'week'] = forecast_weeks[i]/52
	print(x[['week','num_orders_shift1','num_orders_shift2']],flush=True)
	forecast_num_orders[i] = xgbmodel.predict(x)[0]
	
forecast_num_orders = [None for i in range(total_previous_weeks)] + [previous_num_orders[total_previous_weeks]] + forecast_num_orders
plot_df = pd.DataFrame({'Week': all_weeks, 'Previous Orders': previous_num_orders, 'Forecasted Orders': forecast_num_orders})
st.line_chart(plot_df,x='Week',y=['Previous Orders','Forecasted Orders'])


