import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

data = {
    '2017': pd.read_excel(DATA_DIR / 'mancanegara_2017.xlsx'),
    '2018': pd.read_excel(DATA_DIR / 'mancanegara_2018.xlsx'),
    '2019': pd.read_excel(DATA_DIR / 'mancanegara_2019.xlsx'),
    '2020': pd.read_excel(DATA_DIR / 'mancanegara_2020.xlsx'),
    '2021': pd.read_excel(DATA_DIR / 'mancanegara_2021.xlsx'),
    '2022': pd.read_excel(DATA_DIR / 'mancanegara_2022.xlsx'),
    '2023': pd.read_excel(DATA_DIR / 'mancanegara_2023.xlsx'),
    '2024': pd.read_excel(DATA_DIR / 'mancanegara_2024.xlsx'),
}

df = pd.concat(data.values(), ignore_index=True)
df = df.drop(columns=['type','yearly'])
years = []
for year, df_year in data.items():
    years.extend([year] * len(df_year))

# Add the 'year' column to the DataFrame
df['year'] = years

for col in df.columns:
  if col != 'entrance':
    # Replace hyphens or similar characters with NaN before conversion
    df[col] = df[col].replace('-', np.nan)
    # Convert the column to numeric, handling errors
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

df = df.drop(['entrance'], axis=1)

# Group data by year and sum the numerical columns
df_grouped = df.groupby('year').sum()
df_grouped = df_grouped[1:]

# with st.container():
#     st.write("### **Data Wisatawan Mancanegara**")
#     st.write(df_grouped)
#     st.write('---')

months = range(1, 13)

new_data = []

for year, row in df_grouped.iterrows():
    for month in months:
        value = row.iloc[month - 1]
        new_data.append([f"{year}-{month}", value])

new_df = pd.DataFrame(
    new_data,
    columns=['Period', 'Value']
)

new_df['Period'] = pd.to_datetime(
    new_df['Period'],
    format='%Y-%m'
)

new_df = new_df[:-1]

# Data Devisa
pengeluaran = {'Tahun': [2018, 2019, 2020, 2021, 2022, 2023, 2024], 'Rata-Rata Pengeluaran': [1220.18, 1154.64, 2165.02, 3097.41, 1448.01, 1625.36, 1383.78]}
pengeluaran_df = pd.DataFrame(pengeluaran)
merged_df = pd.merge(new_df, pengeluaran_df, left_on=new_df['Period'].dt.year, right_on='Tahun', how='left')

# Multiply 'Value' by 'Rata-Rata Pengeluaran'
merged_df['Devisa'] = merged_df['Value'] * merged_df['Rata-Rata Pengeluaran']
merged_df = merged_df[['Period', 'Devisa']]

# Box-Cox
data = new_df.copy()
data['Value'] = data['Value'].apply(lambda x: x if x > 0 else 1e-6)
data['Value_Boxcox'], lam = boxcox(data['Value'])

# Box-Cox
devisa = merged_df.copy()
devisa['Devisa'] = devisa['Devisa'].apply(lambda x: x if x > 0 else 1e-6)
devisa['Devisa_Boxcox'], lamd = boxcox(devisa['Devisa'])

# Difference the data
data["Value_diff"] = data["Value_Boxcox"].diff()
data.dropna(inplace=True)

devisa["Devisa_diff"] = devisa["Devisa_Boxcox"].diff()
devisa.dropna(inplace=True)

from statsmodels.tsa.statespace.sarimax import SARIMAX

data['Period'] = pd.to_datetime(data['Period'])
data['month_num'] = data['Period'].dt.month

# Get fourier features
for order in range(1, 21):
    data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * data['month_num'] / 12)
    data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * data['month_num'] / 12)

# name of fourier features
fourier_features = [i for i in list(data) if i.startswith('fourier')]

devisa['Period'] = pd.to_datetime(devisa['Period'])
devisa['month_num'] = devisa['Period'].dt.month

# Get fourier features
for order in range(1, 21):
    devisa[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * devisa['month_num'] / 12)
    devisa[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * devisa['month_num'] / 12)

# name of fourier features
fourier_features_d = [i for i in list(devisa) if i.startswith('fourier')]

try:
    with open(BASE_DIR / 'wisman_model.pkl', 'rb') as file:
        modelw = pickle.load(file)

    st.success("Wisman model loaded successfully")

    with open(BASE_DIR / 'devisa_model.pkl', 'rb') as file:
        modeld = pickle.load(file)

    st.success("Devisa model loaded successfully")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.exception(e)
    st.stop()

def forecast_wisatawan(months):
    future_dates = pd.date_range(start=data['Period'].max(), periods=months + 1, freq='MS')[1:]
    future_data = pd.DataFrame({'Period': future_dates})
    future_data['month_num'] = future_data['Period'].dt.month
    
    # Generate Fourier features
    for order in range(1, 21):
        future_data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * future_data['month_num'] / 12)
        future_data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * future_data['month_num'] / 12)
    
    # Ubah ke NumPy array (.values) untuk menghindari error indeks pandas di statsmodels
    exog_future = future_data[fourier_features].values
    
    # Forecast
    boxcox_future_forecasts = modelw.forecast(months, exog=exog_future)
    future_forecasts = inv_boxcox(boxcox_future_forecasts, lam)
    
    return future_dates, future_forecasts

def forecast_devisa(months):
    future_dates = pd.date_range(start=devisa['Period'].max(), periods=months + 1, freq='MS')[1:]
    future_data_d = pd.DataFrame({'Period': future_dates})
    future_data_d['month_num'] = future_data_d['Period'].dt.month
    
    # Generate Fourier features
    for order in range(1, 21):
        future_data_d[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * future_data_d['month_num'] / 12)
        future_data_d[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * future_data_d['month_num'] / 12)
    
    # Ubah ke NumPy array (.values) untuk menghindari error indeks pandas di statsmodels
    exog_future_d = future_data_d[fourier_features_d].values
    
    # Forecast
    boxcox_future_forecasts_d = modeld.forecast(months, exog=exog_future_d)
    future_forecasts_d = inv_boxcox(boxcox_future_forecasts_d, lamd)
    
    return future_dates, future_forecasts_d

# Streamlit UI
st.set_page_config(page_title='Dashboard Wisatawan Mancanegara', page_icon=':airplane:', layout='wide')

with st.container():

    st.write('''
             # Forecast Wisatawan Mancanegara
             ''')
    st.write("---")

pilihan = st.selectbox('Menu: ', ['','🗃️View Data', '📈Forecast'])

try: 
    if 'View Data' in pilihan:
        with st.container():
            st.write("### **Data Wisatawan Mancanegara**")
            st.write(df_grouped)
            st.write('---')

    elif 'Forecast' in pilihan: 
        st.header("Input Parameters")
        months_to_forecast = st.number_input("Months to Forecast", min_value=0)
        if st.button("Generate Forecast"):
            future_dates, future_values = forecast_wisatawan(months_to_forecast)
            
            future_df = pd.DataFrame({'Period': future_dates, 'Forecast': future_values})
            def persentase_selisih(n_awal, n_akhir):
                return ((n_akhir - n_awal) / n_awal) * 100

            a = new_df['Value'].iloc[-12:].mean()
            b = future_values.mean()

            persentase = persentase_selisih(a, b)

            if persentase > 0:
                st.write(f'📈Terjadi peningkatan sebesar {persentase:.2f}% dari periode sebelumnya')
            elif persentase < 0:
                st.write(f'📉Terjadi penurunan sebesar {persentase:.2f}% dari periode sebelumnya')
            else:
                st.write(f'📊Tidak ada peningkatan maupun penurunan dari periode sebelumnya')

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Period'], y=data['Value'], name='Actual Data', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_values, name='Forecast', line=dict(color='orange')))
            
            fig.update_layout(template="simple_white", font=dict(size=18), title_text='Forecast',
                            width=900, title_x=0.5, height=400, xaxis_title='Date',
                            yaxis_title='Wisatawan')
            
            st.plotly_chart(fig)

            future_dates, future_devisa = forecast_devisa(months_to_forecast)
            a = devisa['Devisa'].iloc[-12:].mean()
            b = future_devisa.mean()

            persentase = persentase_selisih(a, b)

            if persentase > 0:
                st.write(f'📈Terjadi peningkatan sebesar {persentase:.2f}% dari periode sebelumnya')
            elif persentase < 0:
                st.write(f'📉Terjadi penurunan sebesar {persentase:.2f}% dari periode sebelumnya')
            else:
                st.write(f'📊Tidak ada peningkatan maupun penurunan dari periode sebelumnya')

            # Plot
            figd = go.Figure()
            figd.add_trace(go.Scatter(x=devisa['Period'], y=devisa['Devisa'], name='Actual Data', line=dict(color='blue')))
            figd.add_trace(go.Scatter(x=future_dates, y=future_devisa, name='Forecast', line=dict(color='orange')))
            
            figd.update_layout(template="simple_white", font=dict(size=18), title_text='Forecast',
                            width=900, title_x=0.5, height=400, xaxis_title='Date',
                            yaxis_title='Devisa')
            
            st.plotly_chart(figd)
            

    else:
        print('')

except Exception as e:
    st.error(f"Error saat memuat model atau data: {e}")

