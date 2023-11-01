# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:34:03 2023

@author: Nick
"""
import pandas as pd
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt

#Read in time series forecast data
time_series=pd.read_csv('forecast_prophet_data.csv')

#Load min max scaler and the model
min_max_scaler=joblib.load('min_max_scaler_grid.joblib')
rf_model=joblib.load('rf_grid_best_model.pkl')

files=os.listdir()
pkl_files=[file for file in files if file.endswith('.pkl')]

#Get all the columns that need encoded
columns_encoded=['age_group','sex','race','ethnicity','exposure_yn','hosp_yn','icu_yn','underlying_conditions_yn','month']

#Set up empty df
user_input={'month':[],
            'age_group':[],
            'sex':[],
            'race':[],
            'ethnicity':[],
            'case_positive_specimen_interval':[],
            'case_onset_interval':[],
            'exposure_yn':[],
            'hosp_yn':[],
            'icu_yn':[],
            'underlying_conditions_yn':[],
            'death_yn':[],
            'county_population':[],
            'county_population_density':[]}

user_df=pd.DataFrame(user_input)

#Set data function/initialize
def initialize_state():
    return {'data':None}

session_state=initialize_state()
user_input=initialize_state()

#Structure mortality section
st.title('COVID-19 Mortality Prediction and Forecast Dashboard')

drop_month=st.selectbox('Case Month',['January','February','March','April','May','June','July','August',
                                 'September','October','November','December'])
drop_age=st.selectbox('Age Group',['0 - 17 years','18 to 49 years','50 to 64 years','65+ years'])
drop_sex=st.selectbox('Sex',['Male','Female'])
drop_race=st.selectbox('Race',['American Indian/Alaska Native','Asian','Black','Multiple/Other','Native Hawaiian/Other Pacific Islander',
                               'White'])
drop_ethnicity=st.selectbox('Ethnicity',['Hispanic/Latino','Non-Hispanic/Latino'])
drop_exposure=st.selectbox('Exposure',['Unknown','Yes'])
drop_hosp=st.selectbox('Hospitalized?',['No','Yes'])
drop_icu=st.selectbox('ICU?',['No','Yes'])
drop_conditions=st.selectbox('Underlying Conditions?',['No','Yes'])

num_specimen=st.number_input('Positive Specimen Interval',value=0.0)
num_onset=st.number_input('Case Onset Interval',value=0.0)
num_countypop=st.number_input('Patient County Population',value=0.0)
num_countydensity=st.number_input('Patient County Population Density',value=0.0)

#If the user presses submit, take those inputs, scale/transform, predict, and display prediction to user along with inputs
if st.button('Submit Inputs'):
          
    user_input={'month':[],
                'age_group':[],
                'sex':[],
                'race':[],
                'ethnicity':[],
                'case_positive_specimen_interval':[],
                'case_onset_interval':[],
                'exposure_yn':[],
                'hosp_yn':[],
                'icu_yn':[],
                'underlying_conditions_yn':[],
                'death_yn':[],
                'county_population':[],
                'county_population_density':[]}
    
    user_df=pd.DataFrame(user_input)
    
    case={'month':[drop_month],
          'age_group':[drop_age],
          'sex':[drop_sex],
          'race':[drop_race],
          'ethnicity':[drop_ethnicity],
          'case_positive_specimen_interval':[num_specimen],
          'case_onset_interval':[num_onset],
          'exposure_yn':[drop_exposure],
          'hosp_yn':[drop_hosp],
          'icu_yn':[drop_icu],
          'underlying_conditions_yn':[drop_conditions],
          'death_yn':0,
          'county_population':[num_countypop],
          'county_population_density':[num_countydensity]}
   
    user_df=user_df.append(pd.DataFrame(case),ignore_index=True)
    
    numeric_data=user_df.drop(columns_encoded,axis=1)
    cat_data=user_df[columns_encoded]
    encoded_cat_data=pd.DataFrame()
    
    for i in columns_encoded:
        cat_data_column=pd.DataFrame(cat_data[i])
        pkl_file=[file_a for file_a in pkl_files if i in file_a]
        cat_encoder=joblib.load(pkl_file[0])
        cat_encoded=cat_encoder.fit_transform(cat_data_column)
        feature_names=cat_encoder.get_feature_names_out()
        new_feature_names=[f"{name}" for name in feature_names]
        cat_encoded_df=pd.DataFrame(data=cat_encoded.toarray(),columns=new_feature_names)
        encoded_cat_data=pd.concat([encoded_cat_data,cat_encoded_df],axis=1)
    print(encoded_cat_data)
    
    final_columns=['age_group_0 - 17 years', 'age_group_18 to 49 years',
           'age_group_50 to 64 years', 'age_group_65+ years', 'sex_Female',
           'sex_Male', 'race_American Indian/Alaska Native', 'race_Asian',
           'race_Black', 'race_Multiple/Other',
           'race_Native Hawaiian/Other Pacific Islander', 'race_White',
           'ethnicity_Hispanic/Latino', 'ethnicity_Non-Hispanic/Latino',
           'exposure_yn_Unknown', 'exposure_yn_Yes', 'hosp_yn_No', 'hosp_yn_Yes',
           'icu_yn_No', 'icu_yn_Yes', 'underlying_conditions_yn_No',
           'underlying_conditions_yn_Yes', 'month_April', 'month_August',
           'month_December', 'month_February', 'month_January', 'month_July',
           'month_June', 'month_March', 'month_May', 'month_November',
           'month_October', 'month_September']
    
    final_df=pd.DataFrame(0,columns=final_columns,index=encoded_cat_data.index)
    final_df[encoded_cat_data.columns]=encoded_cat_data
    print(final_df)
    
    scaled_num_data=pd.DataFrame(min_max_scaler.transform(numeric_data))
    scaled_num_data.columns=numeric_data.columns
    
    rf_cat_encoding=pd.concat([final_df,scaled_num_data],axis=1)
    rf_cat_encoding.drop('death_yn',axis=1,inplace=True)
    session_state['data']=rf_cat_encoding
    user_input['data']=user_df
    
    prediction=rf_model.predict(session_state['data'])
    if prediction==0:
        st.write('Prediction: Survivable')

    if prediction==1:
        st.write('Prediction: Fatal')
    
if session_state['data'] is not None:
    st.write('User Input')
    st.dataframe(user_input['data'])

#Display the time series forecast    
forecast_df=pd.DataFrame()
if st.button('Run Forecast'):
   forecast_df['Month']=time_series['month']
   forecast_df['Prior Month Percent Change Percent']=time_series['percent_change']
   forecast_df['Trend from Prior Month']=time_series['change']
   st.write(r'Hospitalization Percent Change (by Month))')
   st.dataframe(forecast_df)
   
   st.write('Monthly Percent Change Graph')
   fig,ax=plt.subplots()
   ax.scatter(forecast_df['Month'],forecast_df['Prior Month Percent Change Percent'],label='Data Points')
   ax.set_xlabel('Month')
   ax.set_ylabel('Percent Change from Prior Month (absolute value)')
   ax.set_xticklabels(forecast_df['Month'],rotation=90)
   st.pyplot(fig)
   
   
    

        
#cd C:\Users\Nick\Desktop\DAAN888\Dashboard