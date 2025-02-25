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
df_grouped = df_grouped[1:]

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

with open('wisman_model.pkl', 'rb') as file:
    modelw = pickle.load(file)

with open('devisa_model.pkl', 'rb') as file:
    modeld = pickle.load(file)

def forecast_wisatawan(months):
    future_dates = pd.date_range(start=data['Period'].max(), periods=months + 1, freq='MS')[1:]
    future_data = pd.DataFrame({'Period': future_dates})
    future_data['month_num'] = future_data['Period'].dt.month
    
    # Generate Fourier features
    for order in range(1, 25):
        future_data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * future_data['month_num'] / 12)
        future_data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * future_data['month_num'] / 12)
    
    # Forecast
    boxcox_future_forecasts = modelw.forecast(months, exog=future_data[fourier_features])
    future_forecasts = inv_boxcox(boxcox_future_forecasts, lam)
    
    return future_data['Period'], future_forecasts

def forecast_devisa(months):
    future_dates = pd.date_range(start=devisa['Period'].max(), periods=months + 1, freq='MS')[1:]
    future_data_d = pd.DataFrame({'Period': future_dates})
    future_data_d['month_num'] = future_data_d['Period'].dt.month
    
    # Generate Fourier features
    for order in range(1, 25):
        future_data_d[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * future_data_d['month_num'] / 12)
        future_data_d[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * future_data_d['month_num'] / 12)
    
    # Forecast
    boxcox_future_forecasts_d = modeld.forecast(months, exog=future_data_d[fourier_features_d])
    future_forecasts_d = inv_boxcox(boxcox_future_forecasts_d, lamd)
    
    return future_data_d['Period'], future_forecasts_d

# Streamlit UI
st.set_page_config(page_title='Dashboard Wisatawan Mancanegara', page_icon=':airplane:', layout='wide')

with st.container():

    st.write('''
             # Forecast Wisatawan Mancanegara
             ''')
    st.write("---")

pilihan = st.selectbox('Menu: ', ['','🗃️View Tourist Data', '📈Forecast'])

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

            persentase_w = persentase_selisih(a, b)

            if persentase_w > 0:
                st.write(f'📈Terjadi peningkatan sebesar {persentase_w:.2f}% dari periode sebelumnya')
            elif persentase_w < 0:
                st.write(f'📉Terjadi penurunan sebesar {persentase_w:.2f}% dari periode sebelumnya')
            else:
                st.write(f'📊Tidak ada peningkatan maupun penurunan dari periode sebelumnya')

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Period'], y=data['Value'], name='Actual Data', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_values, name='Forecast', line=dict(color='orange')))
            
            fig.update_layout(template="simple_white", font=dict(size=18), title_text='Forecast Wisman',
                            width=900, title_x=0.5, height=400, xaxis_title='Date',
                            yaxis_title='Wisatawan')
            
            st.plotly_chart(fig)

            future_dates, future_devisa = forecast_devisa(months_to_forecast)
            a = devisa['Devisa'].iloc[-12:].mean()
            b = future_devisa.mean()

            persentase_d = persentase_selisih(a, b)

            if persentase_d > 0:
                st.write(f'📈Terjadi peningkatan sebesar {persentase_d:.2f}% dari periode sebelumnya')
            elif persentase_d < 0:
                st.write(f'📉Terjadi penurunan sebesar {persentase_d:.2f}% dari periode sebelumnya')
            else:
                st.write(f'📊Tidak ada peningkatan maupun penurunan dari periode sebelumnya')

            # Plot
            figd = go.Figure()
            figd.add_trace(go.Scatter(x=devisa['Period'], y=devisa['Devisa'], name='Actual Data', line=dict(color='blue')))
            figd.add_trace(go.Scatter(x=future_dates, y=future_devisa, name='Forecast', line=dict(color='orange')))
            
            figd.update_layout(template="simple_white", font=dict(size=18), title_text='Forecast Devisa',
                            width=900, title_x=0.5, height=400, xaxis_title='Date',
                            yaxis_title='Devisa')
            
            st.plotly_chart(figd)

            if persentase_w > 0:
                if persentase_d > 0:
                    st.write(f'Terjadi peningkatan pada kunjungan wisatawan mancanegara sebesar {persentase_w:.2f}% dan peningkatan devisa sebesar {persentase_d:.2f}% dari periode sebelumnya.')
                elif persentase_d < 0:
                    st.write(f'Terjadi peningkatan pada kunjungan wisatawan mancanegara sebesar {persentase_w:.2f}%, namun terjadi penurunan devisa sebesar {persentase_d:.2f}% dari periode sebelumnya.')
                else:
                    st.write(f'Terjadi peningkatan pada kunjungan wisatawan mancanegara sebesar {persentase_w:.2f}%, namun tidak ada peningkatan maupun penurunan devisa dari periode sebelumnya.')
            elif persentase_w < 0:
                if persentase_d > 0:
                    st.write(f'Terjadi penurunan pada kunjungan wisatawan mancanegara sebesar {persentase_w:.2f}%, namun terdapat peningkatan devisa sebesar {persentase_d:.2f}% dari periode sebelumnya.')
                elif persentase_d < 0:
                    st.write(f'Terjadi penuurunan pada kunjungan wisatawan mancanegara sebesar {persentase_w:.2f}% dan terjadi penurunan devisa sebesar {persentase_d:.2f}% dari periode sebelumnya.')
                else:
                    st.write(f'Terjadi penurunan pada kunjungan wisatawan mancanegara sebesar {persentase_w:.2f}%, namun tidak ada peningkatan maupun penurunan devisa dari periode sebelumnya.')
            else:
                if persentase_d > 0:
                    st.write(f'Tidak ada peningkatan maupun penurunan kunjungan wisatawan mancanegara dari periode sebelumnya, namun terdapat peningkatan devisa sebesar {persentase_d:.2f}% dari periode sebelumnya.')
                elif persentase_d < 0:
                    st.write(f'Tidak ada peningkatan maupun penurunan kunjungan wisatawan mancanegara dari periode sebelumnya, namun terjadi penurunan devisa sebesar {persentase_d:.2f}% dari periode sebelumnya.')
                else:
                    st.write(f'Tidak ada peningkatan maupun penurunan kunjungan wisatawan mancanegara dan devisa dari periode sebelumnya.')

            st.write('''
            Rekomendasi:
            1. Penguatan Destinasi Wisata
            
            Meningkatkan daya saing destinasi wisata melalui pengembangan fasilitas, perbaikan citra pariwisata, peningkatan kualitas pengelolaan destinasi, serta pemberdayaan masyarakat lokal.
            
            2. Optimalisasi Pemasaran Pariwisata Nasional
            
            Memperluas kerja sama internasional dalam sektor pariwisata serta menarik lebih banyak wisatawan mancanegara guna meningkatkan kunjungan dan devisa negara.
            
            3. Penguatan Industri Pariwisata
            
            Mendorong keterlibatan lebih besar dari pelaku usaha lokal dalam industri pariwisata nasional serta meningkatkan daya saing produk wisata agar lebih menarik dan berkualitas.
            
            4. Peningkatan Kapasitas Kelembagaan
            
            Mengembangkan sumber daya manusia pariwisata serta memperkuat kelembagaan dan organisasi sektor pariwisata untuk mendukung pertumbuhan industri secara berkelanjutan.
            ''')
            

    else:
        print('')

except Exception as e:
    st.error(f"Error saat memuat model atau data: {e}")
