import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('xAPI-Edu-Data.csv')

df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
df = df.sort_values('date')

attendance_data = df.groupby('date')['StudentAbsenceDays'].mean().reset_index()
attendance_data.columns = ['ds', 'y']  

model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(attendance_data)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

result = pd.concat([attendance_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]], axis=1)

result['risk_trend'] = result['yhat'].diff().apply(lambda x: 'Declining' if x < -0.5 else 'Stable/Improving')

result.to_csv('time_series_predictions.csv', index=False)
print("✅ Time series predictions saved to 'time_series_predictions.csv'")
print(result.tail(10))
