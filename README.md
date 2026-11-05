Forecasting International Tourist Arrivals and Tourism Foreign Exchange Revenue
📌 Project Overview

This project is a time series forecasting dashboard developed using Python and Streamlit to predict international tourist arrivals (Wisatawan Mancanegara / Wisman) and tourism foreign exchange revenue (Devisa Pariwisata) in Indonesia.

The application uses historical monthly international tourist arrival data from 2017 to 2024 to analyze tourism trends and generate future predictions. It also estimates tourism foreign exchange revenue based on the average expenditure of international tourists.

The forecasting models are built using SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors), incorporating Fourier features to capture seasonal patterns in tourism data.

📊 Key Results & Achievements
Forecasting Model Development: Developed and evaluated SARIMAX and Dynamic Harmonic Regression models to forecast international tourist arrivals in Indonesia, achieving 94.70% forecasting accuracy (MAPE: 5.30%), with an MAE of 27,730 and RMSE of 34,320.
Tourism Growth Forecast: Projected an average increase of 85,546 international tourist arrivals (7.48%) during the 2024–2026 forecasting period compared to the previous period, providing data-driven insights into Indonesia's international tourism growth trends.
Data-Driven Insights: Translated forecasting results into actionable recommendations to support tourism planning, destination development, and strategies for increasing Indonesia's tourism foreign exchange revenue.

🎯 Project Objectives
Analyze historical international tourist arrival data in Indonesia.
Forecast future international tourist arrivals based on historical trends and seasonal patterns.
Estimate future tourism foreign exchange revenue.
Compare forecast results with historical performance.
Provide insights and recommendations to support tourism development and decision-making.
🛠️ Technologies & Libraries

This project was developed using the following technologies:

Python – Main programming language.
Pandas – Data processing, manipulation, and aggregation.
NumPy – Numerical computations and Fourier feature generation.
SciPy – Box-Cox transformation and inverse transformation.
Statsmodels – SARIMAX time series forecasting.
Plotly – Interactive data visualization and forecasting charts.
Streamlit – Interactive web dashboard development.
Pickle – Loading pre-trained forecasting models.

📊 Methodology

The forecasting process consists of several stages:

Data Collection
Collect monthly international tourist arrival data from 2017 to 2024, stored in Excel files.
Data Preprocessing
Combine datasets from multiple years.
Handle missing and non-numeric values.
Convert data into numerical format.
Aggregate tourist arrivals by year and month.
Prepare monthly time series data.
Tourism Foreign Exchange Calculation
Estimate foreign exchange revenue by multiplying the number of international tourist arrivals by the average expenditure per tourist.

Formula:

Devisa = Tourist Arrivals × Average Tourist Expenditure

Box-Cox Transformation
Apply Box-Cox transformation to stabilize variance and improve the characteristics of the time series data.
Differencing
Apply first-order differencing to help achieve stationarity.
Seasonality Modeling with Fourier Features
Generate sine and cosine features to capture recurring monthly seasonal patterns.
SARIMAX Forecasting
Use pre-trained SARIMAX models to forecast:
International tourist arrivals.
Tourism foreign exchange revenue.
Inverse Transformation
Convert forecasted values back to their original scale using the inverse Box-Cox transformation.
Visualization and Interpretation
Display historical data, forecasting results, percentage changes, and recommendations through an interactive dashboard.
✨ Dashboard Features

The Streamlit application provides two main features:

1. View Tourist Data
Display aggregated international tourist arrival data.
Present historical data for the period 2017–2024.
2. Forecast
Allow users to specify the number of months to forecast.
Generate predictions for international tourist arrivals.
Generate predictions for tourism foreign exchange revenue.
Visualize historical and forecasted values using Plotly charts.
Calculate percentage increases or decreases compared to the previous 12-month average.
Provide tourism development recommendations based on forecast results.

📁 Project Structure
text
├── app.py
├── mancanegara_2017.xlsx
├── mancanegara_2018.xlsx
├── mancanegara_2019.xlsx
├── mancanegara_2020.xlsx
├── mancanegara_2021.xlsx
├── mancanegara_2022.xlsx
├── mancanegara_2023.xlsx
├── mancanegara_2024.xlsx
├── wisman_model.pkl
├── devisa_model.pkl
└── README.md

🚀 Installation and Usage
1. Clone the Repository
bash
git clone https://github.com/yusufnasihin/dhr-wisman-dashboard.git

2. Navigate to the Project Directory
bash
cd your-repository

3. Install Dependencies
bash
pip install pandas numpy scipy statsmodels plotly streamlit openpyxl

4. Run the Streamlit Application
bash
streamlit run app.py

5. Access the Dashboard

Open the local URL provided by Streamlit, usually:

text
http://localhost:8501

📈 Forecasting Output

The dashboard generates two main forecasting visualizations:

International Tourist Arrivals Forecast: Predicts the number of international tourists visiting Indonesia.
Tourism Foreign Exchange Revenue Forecast: Estimates potential foreign exchange revenue generated from international tourist expenditure.

Both visualizations compare historical observations with future predictions and provide percentage-based trend interpretations.

💡 Recommendations

Based on the forecasting results, the application provides several strategic recommendations:

Tourism Destination Development – Improve tourism facilities, destination management, and local community involvement.
National Tourism Marketing Optimization – Expand international partnerships and strengthen tourism promotion.
Tourism Industry Strengthening – Encourage local business participation and improve the competitiveness of tourism products.
Institutional Capacity Development – Improve tourism human resources and strengthen institutional collaboration.
⚠️ Notes
The forecasting application requires the Excel datasets and pre-trained model files (.pkl) to be available in the project directory.
Forecasting accuracy depends on the quality of historical data and the performance of the trained SARIMAX models.
Tourism foreign exchange revenue is an estimate based on average tourist expenditure, rather than actual recorded government revenue.
The application uses pre-trained models and does not perform model training during dashboard execution.
👨‍💻 Author

Developed as a data analytics and time series forecasting project focused on international tourism in Indonesia.
