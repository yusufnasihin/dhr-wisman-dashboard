import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle

data = {
    '2017': pd.read_excel('mancanegara_2017.xlsx'),
    '2018': pd.read_excel('mancanegara_2018.xlsx'),
    '2019': pd.read_excel('mancanegara_2019.xlsx'),
    '2020': pd.read_excel('mancanegara_2020.xlsx'),
    '2021': pd.read_excel('mancanegara_2021.xlsx'),
    '2022': pd.read_excel('mancanegara_2022.xlsx'),
    '2023': pd.read_excel('mancanegara_2023.xlsx'),
    '2024': pd.read_excel('mancanegara_2024.xlsx'),
}

df = pd.concat(data.values(), ignore_index=True)
df = df.drop(columns=['type','yearly'], axis=1)
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

# with st.container():
#     st.write("### **Data Wisatawan Mancanegara**")
#     st.write(df_grouped)
#     st.write('---')

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Create an empty list to store the new data
new_data = []

# Iterate through each row of the DataFrame
for index, row in df_grouped.iterrows():  # Iterate through the grouped DataFrame (df_grouped)
    year = index  # The year is now the index of the grouped DataFrame
    # Iterate through each month
    for month in months:
        # Create a new row with period and value
        # Access the correct column name for the sum of values (e.g., 'Jan')
        # Instead of row[month], use row[month -1] to access the column value

        # Subtract 1 from the month to adjust for zero-based indexing in pandas
        new_data.append([f"{year}-{month}", row[month - 1]])

# Create a new DataFrame from the new data
new_df = pd.DataFrame(new_data, columns=['Period', 'Value'])
new_df['Period'] = pd.to_datetime(new_df['Period'], format='%Y-%m')
new_df = new_df[:-1]

data = new_df.copy()
data['Value'] = data['Value'].apply(lambda x: x if x > 0 else 1e-6)
data['Value_Boxcox'], lam = boxcox(data['Value'])

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Difference the data
data["Value_diff"] = data["Value_Boxcox"].diff()
data.dropna(inplace=True)

data['Period'] = pd.to_datetime(data['Period'])
data['month_num'] = data['Period'].dt.month

from statsmodels.tsa.statespace.sarimax import SARIMAX

data['Period'] = pd.to_datetime(data['Period'])
data['month_num'] = data['Period'].dt.month

# Get fourier features
for order in range(1, 13):
    data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * data['month_num'] / 12)
    data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * data['month_num'] / 12)

# name of fourier features
fourier_features = [i for i in list(data) if i.startswith('fourier')]

with open('dhr_model.pkl', 'rb') as file:
    model = pickle.load(file)


def forecast_wisatawan(months):
    future_dates = pd.date_range(start=data['Period'].max(), periods=months + 1, freq='MS')[1:]
    future_data = pd.DataFrame({'Period': future_dates})
    future_data['month_num'] = future_data['Period'].dt.month
    
    # Generate Fourier features
    for order in range(1, 13):
        future_data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * future_data['month_num'] / 12)
        future_data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * future_data['month_num'] / 12)
    
    # Forecast
    boxcox_future_forecasts = model.forecast(months, exog=future_data[fourier_features])
    future_forecasts = inv_boxcox(boxcox_future_forecasts, lam)
    
    return future_data['Period'], future_forecasts

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

            # Contoh penggunaan
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
            

    else:
        print('')

except Exception as e:
    st.error(f"Error saat memuat model atau data: {e}")

