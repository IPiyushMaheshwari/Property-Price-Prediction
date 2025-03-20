import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
app=Flask(__name__) 

# Load and preprocess the data (same as before)
file_path = 'fully_updated_minorprojectteam.xlsx'  # Replace with your file path
sheet_data = pd.read_excel(file_path, sheet_name='Sheet1')

# Data preprocessing
price_data = sheet_data.melt(
    id_vars=["Area", "Landmark", "Category"],
    var_name="Year",
    value_name="Price"
)

price_data["Year"] = pd.to_numeric(price_data["Year"], errors='coerce')
price_data = price_data.dropna(subset=["Year", "Price"])

price_data["Area"] = price_data["Area"].fillna("Unknown")
price_data["Landmark"] = price_data["Landmark"].fillna("Unknown")

price_data = pd.get_dummies(price_data, columns=["Area", "Landmark", "Category"], drop_first=True)

X = price_data.drop(columns=["Price"])
y = price_data["Price"]

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X, y)

# Flask Setup
app = Flask(__name__)

def calculate_growth_trend(year, price_2018, price_2024, max_growth_year=2040):
    growth_rate = (price_2024 - price_2018) / (2024 - 2018)
    predicted_price = price_2024 + growth_rate * (year - 2024)

    if year > max_growth_year:
        max_price = price_2024 + growth_rate * (max_growth_year - 2024)
        predicted_price = min(predicted_price, max_price)

    return predicted_price

@app.route('/')
def one():
    return render_template('propertya(pd).html')

@app.route('/predict', methods=['POST'])
def predict():
    area = request.form.get('area')
    landmark = request.form.get('landmark')
    year = int(request.form.get('year'))
    
    area_column = f"Area_{area}"
    landmark_column = f"Landmark_{landmark}"
    
    area_data = price_data[(price_data[area_column] == 1) & (price_data[landmark_column] == 1)]
    
    price_2018 = area_data[area_data["Year"] == 2018]["Price"].values
    price_2024 = area_data[area_data["Year"] == 2024]["Price"].values
    
    if price_2018.size == 0 or price_2024.size == 0:
        return f"No data available for {area} at {landmark} for years 2018 and 2024."

    future_price = calculate_growth_trend(year, price_2018[0], price_2024[0])

    # Create a list of years to plot the growth trend
    years = np.arange(2018, year+1)
    predicted_prices = [calculate_growth_trend(y, price_2018[0], price_2024[0]) for y in years]

    # Plot the growth trend using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years, y=predicted_prices, mode='lines+markers', name="Price Trend", line=dict(color="blue")
    ))

    fig.add_vline(x=year, line=dict(color="red", dash="dash"), name=f"Prediction for {year}")
    
    fig.update_layout(
        title=f"Price Growth Trend for {area} at {landmark}",
        xaxis_title="Year",
        yaxis_title="Price",
        showlegend=True
    )

    graph_html = fig.to_html(full_html=False)
    
    return render_template('propertya(pe).html', predicted_price=f"Predicted price for {area} at {landmark} in {year}: {future_price:.2f}", graph_html=graph_html)
    
if __name__ == '__main__':
    app.run(debug=True)


if __name__=="__main__":
    app.run(debug=True)
