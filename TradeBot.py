from flask import Flask, request, jsonify,session,redirect,render_template,send_file
import pandas as pd
from datetime import datetime
import dash_table
import requests
from flask_caching import Cache
app = Flask(__name__)

from bs4 import BeautifulSoup
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.cached(timeout=86400)
def get_news():
    # Replace 'YOUR_API_KEY' with your actual News API key
    api_key = '405bf4b38a45452bacf13d852eea0619'
    l = []
    base_url = 'https://newsapi.org/v2/everything'

    # List of stock symbols
    stock_symbols = [
        'CIPLA', 'INDUSINDBK', 'TECHM', 'WIPRO', 'BHARTIARTL',
        'AXISBANK', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'RELIANCE',
        'LTIM', 'TCS', 'LT', 'INFY', 'SBILIFE', 'HINDALCO',
        'KOTAKBANK', 'TATASTEEL', 'UPL', 'ONGC'
    ]

    for symbol in stock_symbols:
        try:
            params = {
                'q': f'{symbol} stock',
                'apiKey': api_key,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 1
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get('articles'):
                for article in data['articles']:
                    response = requests.get(article['url'])
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')

                        main_image = soup.find('meta', {'property': 'og:image'})
                        main_image_url = main_image.get('content') if main_image else None

                        headline = soup.find('meta', {'property': 'og:title'})
                        headline_text = headline.get('content') if headline else None

                        l.append({
                            'url': article['url'],
                            'main_image_url': main_image_url,
                            'headline_text': headline_text
                        })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")

    return l



               


        
def get_formatted_date(date_string):
    if len(date_string) == 10:
        # Date string is in the format 'YYYY-MM-DD'
        date_format = "%Y-%m-%d"
    else:
        # Date string is in the format 'YYYY-MM-DD HH:mm:ss'
        date_format = "%Y-%m-%d %H:%M:%S"
    
    date_obj = datetime.strptime(date_string, date_format)
    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date

r'''
def forecast_price(symbol,duration):
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import load_model
    import pandas as pd
    import numpy as np

    data=pd.read_csv(r'C:\Users\DELL\Downloads\stock_data-20231013T115612Z-001\stock_data\{}.csv'.format(symbol))
    model = load_model(r"C:\Users\DELL\Downloads\best_model_attention_-20231013T133012Z-001\best_model_attention_\{}.h5".format(symbol))
    # Select relevant columns
    last_date = data['Date'].iloc[-1]    
    data = data[['Open Price', 'High Price', 'Low Price', 'Close Price', 'Total Traded Quantity', 'Total Traded Value', 'VWAP']]
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    sequence_length=29
    # Select the last 30 days of data for prediction
    last_30_days = data_scaled[-sequence_length:]

    last_date = pd.to_datetime(last_date)

    # Use the trained model to make predictions for the next 10 days
    predicted_sequence = []
    for _ in range(duration):
        predicted_value = model.predict(last_30_days.reshape(1, sequence_length, data_scaled.shape[1]))
        predicted_sequence.append(predicted_value)
        last_30_days = np.concatenate((last_30_days[1:], predicted_value), axis=0)

    # Inverse transform the predicted sequence to get the original scale

    
    response_text = "The Forecast for " + symbol + " is:\n\n"
    
    for values in predicted_sequence:
        next_date = last_date + pd.Timedelta(days=1)
        last_date = next_date


        predicted_sequence_original = scaler.inverse_transform(values)


        # Construct the response text
        response_text += f"Date: {next_date.strftime('%Y-%m-%d')}\n"
        response_text += f"Predicted High Price: {round(predicted_sequence_original[0][1],2):.2f}\n"
        response_text += f"Predicted Open Price: {round(predicted_sequence_original[0][0],2):.2f}\n"
        response_text += f"Predicted Low Price: {round(predicted_sequence_original[0][2],2):.2f}\n"
        response_text += f"Predicted Close Price: {round(predicted_sequence_original[0][3],2):.2f}\n\n"
        print(response_text)
    return response_text
r'''

def forecast_price(symbol, duration):
    import pandas as pd
    data = pd.read_csv(r"C:\Users\agnib\Downloads\predictions-20231109T130013Z-001\predictions\{}.csv".format(symbol))

    response_text = "The Forecast for " + symbol + " is:\n\n"
    for i in range(duration):
        response_text += f"Date: {data['Date'][i]}\n"
        response_text += f"Predicted High Price: {round(data['High Price'][i],2):.2f}\n"
        response_text += f"Predicted Open Price: {round(data['Open Price'][i],2):.2f}\n"
        response_text += f"Predicted Low Price: {round(data['Low Price'][i],2):.2f}\n"
        response_text += f"Predicted Close Price: {round(data['Close Price'][i],2):.2f}\n\n"
        print(response_text)
    return response_text



def stock_price(symbol):
    file_path = r"C:\Users\agnib\Downloads\stock_data-20231109T125620Z-001\stock_data\{}.csv".format(symbol)   
    df = pd.read_csv(file_path)
    last_date = (df["Date"].iloc[-1])

    # Extract the corresponding values
    open_price = df[df["Date"] == last_date]["Open Price"].iloc[0]
    close_price = df[df["Date"] == last_date]["Close Price"].iloc[0]
    high_price = df[df["Date"] == last_date]["High Price"].iloc[0]
    low_price = df[df["Date"] == last_date]["Low Price"].iloc[0]
    total_traded_quantity = df[df["Date"] == last_date]["Total Traded Quantity"].iloc[0]

    response_text = """
    As of {}, the following information is available:
    
        1. Open price: {}
        2. Close price: {}
        3. High price: {}
        4. Low price: {}
        5. Volume: {}
   
    """.format(get_formatted_date(last_date), open_price, close_price, high_price, low_price, total_traded_quantity)
    return response_text

def stock_recommendation(symbol):
    file_path = r"C:\Users\agnib\Downloads\stock_data-20231109T125620Z-001\stock_data\{}.csv".format(symbol)   
    df = pd.read_csv(file_path)
    cols=["RSI Recommendation","Bollinger Recommendation","Stochastic Recommendation","MACD Recommendation","MA Recommendation"]
    buy=0
    sell=0
    hold=0
    for col in cols:
        if df[col].iloc[-1]=="Buy":
            buy=buy+1
        elif df[col].iloc[-1]=="Sell":
            sell=sell+1
        if df[col].iloc[-1]=="Hold":
            hold=hold+1                        
    
    response_text = """Based on our indicators for the stock {}:
    
        1. Buy: {}%
        2. Sell: {}%
        3. Hold: {}%
    """.format(symbol,int(buy*100/5),int(sell*100/5),int(hold*100/5))
    print(response_text)
    return response_text



from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, Response
from dash import Dash
from dash.dependencies import Input, Output
import pandas as pd


import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from pathlib import Path



stock_data = {}
data_folder = Path(r"C:\Users\agnib\Downloads\stock_data-20231109T125620Z-001\stock_data")  # Replace with your folder path
for file_path in data_folder.glob("*.csv"):
    stock_name = file_path.stem
    stock_data[stock_name] = pd.read_csv(file_path)
    stock_data[stock_name]['Date'] = pd.to_datetime(stock_data[stock_name]['Date'])  # Convert 'Date' column to datetime
# Load stock data from CSV files into a dictionary


# Function to calculate percentage change between second last and last 'Close Price'
def calculate_percentage_change(data):
    if len(data) >= 2:
        second_last_close = data.iloc[-2]['Close Price']
        last_close = data.iloc[-1]['Close Price']
        percent_change = ((last_close - second_last_close) / second_last_close) * 100
        return percent_change
    else:
        return None

# Function to generate table for top gainers based on percentage change
def generate_gainers_table(stock_data):
    gainers_data = []

    for stock_name, data in stock_data.items():
        percent_change = calculate_percentage_change(data)
        if percent_change is not None and percent_change > 0:
            gainers_data.append({
                'SYMBOL': stock_name,
                'LTP': data.iloc[-1]['Close Price'],
                '%CHNG': percent_change,
                'VOLUME': data.iloc[-1]['Total Traded Quantity']
            })

    gainers_df = pd.DataFrame(gainers_data)
    gainers_df = gainers_df.sort_values(by='%CHNG', ascending=False).head(5)
    return gainers_df

# Function to generate table for top losers based on percentage change
def generate_losers_table(stock_data):
    losers_data = []

    for stock_name, data in stock_data.items():
        percent_change = calculate_percentage_change(data)
        if percent_change is not None and percent_change < 0:
            losers_data.append({
                'SYMBOL': stock_name,
                'LTP': data.iloc[-1]['Close Price'],
                '%CHNG': percent_change,
                'VOLUME': data.iloc[-1]['Total Traded Quantity']
            })

    losers_df = pd.DataFrame(losers_data)
    losers_df = losers_df.sort_values(by='%CHNG').head(5)
    return losers_df

# Initialize Dash app
dash1_app = dash.Dash(__name__, server=app, url_base_pathname='/dash1-app/')
dash1_app.layout = html.Div(style={'width': '50%', 'margin': 'auto'}, children=[
    
    dcc.Tabs([

        dcc.Tab(label='Gainers', children=[
            html.Div([
                html.H3("Top 5 Gainers"),
                dash_table.DataTable(
                    columns=[
                        {'name': 'SYMBOL', 'id': 'SYMBOL'},
                        {'name': 'LTP', 'id': 'LTP', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': '%CHNG', 'id': '%CHNG', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'VOLUME', 'id': 'VOLUME', 'type': 'numeric', 'format': {'specifier': '.0f'}}
                    ],
                    data=generate_gainers_table(stock_data).to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold', 'color': 'white'},
                    style_data_conditional=[
                        {'if': {'column_id': '%CHNG'},
                         'backgroundColor': '#3D9970', 'color': 'white', 'fontWeight': 'bold'},
                        {'if': {'column_id': '%CHNG', 'filter_query': '{%CHNG} < 0'},
                         'backgroundColor': '#FF4136', 'color': 'white', 'fontWeight': 'bold'}
                    ],
                    style_table={'overflowX': 'scroll'}
                )
            ])
        ]),
        dcc.Tab(label='Losers', children=[
            html.Div([
                html.H3("Top 5 Losers"),
                dash_table.DataTable(
                    columns=[
                        {'name': 'SYMBOL', 'id': 'SYMBOL'},
                        {'name': 'LTP', 'id': 'LTP', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': '%CHNG', 'id': '%CHNG', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'VOLUME', 'id': 'VOLUME', 'type': 'numeric', 'format': {'specifier': '.0f'}}
                    ],
                    data=generate_losers_table(stock_data).to_dict('records'),
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold', 'color': 'white'},
                    style_data_conditional=[
                        {'if': {'column_id': '%CHNG'},
                         'backgroundColor': '#FF4136', 'color': 'white', 'fontWeight': 'bold'}
                    ],
                    style_table={'overflowX': 'scroll'}
                )
            ])
        ]),
    ]),
])

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash-app/')

# Dark mode CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app.css.append_css({"external_url": external_stylesheets})
dash_app.layout = html.Div([
    html.H1("Stock Data Dashboard"),
    html.Div([
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': stock, 'value': stock} for stock in stock_data.keys()],
            value=list(stock_data.keys())[0]  # Default value
        ),
        dcc.Tabs(
            id='duration-tabs',
            value='1m',
            children=[
                dcc.Tab(label='1M', value='1m'),
                dcc.Tab(label='3M', value='3m'),
                dcc.Tab(label='6M', value='6m'),
                dcc.Tab(label='1Yr', value='1y'),
                dcc.Tab(label='5Yr', value='5y')
            ]
        )
    ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginBottom': '20px'}),
    
    html.Div([
        dcc.Graph(
            id='stock-graph',
            config={
                'modeBarButtons': [['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                                   ['toggleSpikelines']]
            }
        ),
        dcc.Dropdown(
            id='graph-type-dropdown',
            options=[
                {'label': 'Candlestick', 'value': 'candlestick'},
                {'label': 'Line', 'value': 'line'},
                {'label': 'OHLC', 'value': 'ohlc'},
                {'label': 'Bar', 'value': 'bar'},
                {'label': 'Area', 'value': 'area'}
                # Add more options for different graph types here
            ],
            value='candlestick',  # Default value
            style={'width': '200px', 'margin': '10px','position': 'absolute', 'top': 30, 'right': -5}
        ),
    ], style={'width': '100%', 'display': 'inline-block', 'marginRight': '20px'}),
    
    html.Div([
        html.Button("Toggle Dark Mode", id="theme-button", n_clicks=0)
    ], style={'position': 'absolute', 'top': 10, 'right': 10})
])

# Callback to update the graph based on dropdown and tabs selections
@dash_app.callback(Output('stock-graph', 'figure'),
              [Input('stock-dropdown', 'value'),
               Input('duration-tabs', 'value'),
               Input('theme-button', 'n_clicks'),
               Input('graph-type-dropdown', 'value')])
def update_graph(selected_stock, selected_duration, n_clicks, selected_graph_type):
    stock_df = stock_data[selected_stock]

    # Define time delta based on selected duration
    if selected_duration == '1m':
        time_delta = pd.DateOffset(months=1)
    elif selected_duration == '3m':
        time_delta = pd.DateOffset(months=3)
    elif selected_duration == '6m':
        time_delta = pd.DateOffset(months=6)
    elif selected_duration == '1y':
        time_delta = pd.DateOffset(years=1)
    elif selected_duration == '5y':
        time_delta = pd.DateOffset(years=5)
    else:
        time_delta = pd.DateOffset(months=1)  # Default to 1 month if invalid value
    
    # Filter data based on selected duration
    filtered_data = stock_df.loc[stock_df['Date'] > (stock_df['Date'].max() - time_delta)]

    # Create Plotly figure based on selected graph type
    fig = go.Figure()
    if selected_graph_type == 'candlestick':
        fig.add_trace(go.Candlestick(
            x=filtered_data['Date'],
            open=filtered_data['Open Price'],
            high=filtered_data['High Price'],
            low=filtered_data['Low Price'],
            close=filtered_data['Close Price'],
            name=selected_stock
        ))
    elif selected_graph_type == 'line':
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Close Price'],
            mode='lines',
            name=selected_stock
        ))
    elif selected_graph_type == 'ohlc':
        fig.add_trace(go.Ohlc(
            x=filtered_data['Date'],
            open=filtered_data['Open Price'],
            high=filtered_data['High Price'],
            low=filtered_data['Low Price'],
            close=filtered_data['Close Price'],
            name=selected_stock
        ))
    elif selected_graph_type == 'bar':
        fig.add_trace(go.Bar(
            x=filtered_data['Date'],
            y=filtered_data['Close Price'],
            name=selected_stock
        ))
    elif selected_graph_type == 'area':
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Close Price'],
            mode='lines',
            fill='tozeroy',
            name=selected_stock
        ))

    fig.update_layout(title=f'{selected_stock} Stock Price',
                      xaxis_title='Date',
                      yaxis_title='Price')
    
    if n_clicks % 2 == 0:
        fig.update_layout(template='plotly_white')
    else:
        fig.update_layout(template='plotly_dark')

    return fig



icon="None"



intent_display_name = None



@app.route('/', methods=['GET'])
def index():
    # Render the template with the collected data
    return render_template('index.html')

@app.route('/news', methods=['GET'])
def news():
    # Get news data using the get_news function
    news_data = get_news()
    # Render the template with the collected data
    return render_template('news.html', news_data=news_data)



@app.route('/webhook', methods=['POST'])
def webhook_handler():
    global intent_display_name
    data = request.get_json()

    intent_display_name = data['queryResult']['intent']['displayName']
    print(intent_display_name)
    if intent_display_name == "Forecast_price":
        if data['queryResult']['parameters']['duration']['unit'] == "mo":
            duration = data['queryResult']['parameters']['duration']['amount'] * 30
            response_text = forecast_price(data['queryResult']['parameters']['trade-bot'], int(duration))
        elif data['queryResult']['parameters']['duration']['unit'] == "yr":
            duration = data['queryResult']['parameters']['duration']['amount'] * 365
            response_text = forecast_price(data['queryResult']['parameters']['trade-bot'], int(duration))
        else:
            response_text = forecast_price(data['queryResult']['parameters']['trade-bot'], int(data['queryResult']['parameters']['duration']['amount']))

        response = {
                        "fulfillmentMessages": [
                            {
                                "text": {
                                    "text": [
                                        response_text
                                    ]
                                }
                            },
                            {
                            'payload': {
                "richContent": [
                    [
                    {
                        "type": "chips",
                        "options": [
                        {
                            "text": "Give Stock Prices"
                        },
                        {
                            "text": "Display Charts/Graphs"
                        }
                        ]
                    }
                    ]
                ]
                }
                        }
                            
                        ]
                    }

        return jsonify(response)



    elif intent_display_name == "Stock Price Inquiry":
        symbol = data['queryResult']['parameters']['trade-bot']
        response_text = stock_price(symbol)
        response = {
                    "fulfillmentMessages": [
                        {
                            "text": {
                                "text": [
                                    response_text
                                ]
                            }
                        },
                        {
                        'payload': {
            "richContent": [
                [
                {
                    "type": "chips",
                    "options": [
                    {
                        "text": "Forecast of Stock Prices"
                    },
                    {
                        "text": "Display Charts/Graphs"
                    }
                    ]
                }
                ]
            ]
            }
                    }
                        
                    ]
                }

        return jsonify(response)
    
    elif intent_display_name == "Stock Price Display":
           # Assuming you have extracted the stock symbol
        stock_symbol = data['queryResult']['parameters']['Trade-Bot']
        global icon
        icon= stock_symbol

      
    # Redirect the user to the stock chart route with the stock symbol as a query parameter
        return redirect('/plot')
    
    elif intent_display_name == "which one to buy":
        symbols= ['CIPLA','INDUSINDBK','TECHM','WIPRO','BHARTIARTL','AXISBANK',
          'HCLTECH','HDFCBANK','HDFCLIFE','RELIANCE','LTIM','TCS','LT','INFY','SBILIFE',
          'HINDALCO','KOTAKBANK','TATASTEEL','UPL','ONGC']
        l=[]
        for symbol in symbols:
            df= pd.read_csv(r"C:\Users\agnib\Downloads\stock_data-20231109T125620Z-001\stock_data\{}.csv".format(symbol)) 

            cols=["RSI Recommendation","Bollinger Recommendation","Stochastic Recommendation","MACD Recommendation","MA Recommendation"]
            buy=0
            

            for col in cols:
                if df[col].iloc[-1]=="Buy":
                    buy=buy+1
            l.append(buy)
        prob= int(max(l)*100/5)    
        max_indices = [i for i, value in enumerate(l) if value == max(l)]
        p=[]
        for i in max_indices:
            p.append(symbols[i])

        response_text = """Based on our indicators today's best buy is/are {} with a probability of {}%.""".format(p,prob)
        print(response_text)
        response= {
                    "fulfillmentMessages": [
                        {
                            "text": {
                                "text": [
                                    response_text
                                ]
                            }
                        }]}
        return jsonify(response)

    elif intent_display_name == "Stock Price Recommendation":
           # Assuming you have extracted the stock symbol
        stock_symbol = data['queryResult']['parameters']['trade-bot']
        response_text=stock_recommendation(stock_symbol)
        response = {
                    "fulfillmentMessages": [
                        {
                            "text": {
                                "text": [
                                    response_text
                                ]
                            }
                        },
                        {
                        'payload': {
            "richContent": [
                [
                {
                    "type": "chips",
                    "options": [
                    {
                        "text": "Forecast of Stock Prices"
                    },
                    {
                        "text": "Display Charts/Graphs"
                    }
                    ]
                }
                ]
            ]
            }
                    }
                        
                    ]
                }

        return jsonify(response)

      
    # Redirect the user to the stock chart route with the stock symbol as a query parameter
        return redirect('/plot')        


@app.route('/plot')
def display_stock_chart():
    global icon
    print(icon)
    file_path = r"C:\Users\agnib\Downloads\stock_data-20231109T125620Z-001\stock_data\{}.csv".format(icon)
    df = pd.read_csv(file_path)
    close_prices = df['Close Price'].tail(100).tolist()

    plt.figure(figsize=(8, 6))
    plt.plot(close_prices)
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title('Close Prices for last 100 days')
  
    # Save the plot as a PNG image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Return the image as a Flask response
    return Response(img_buffer.getvalue(), mimetype='image/png')



    # Here, you would generate the chart using the stock_symbol
    # You can use a charting library or API to do this
@app.route('/image')
def image_logo():
    return send_file(r"C:\Users\agnib\Downloads\Vista Logos\logo-transparent-png.png", mimetype='image/png')



if __name__ == "__main__":
    app.run(debug=True)