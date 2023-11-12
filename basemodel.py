import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mse_list = []
stockList = ["INTC", "PFE", "CMCSA", "SAP", "TMUS", "TMO", "AMD", "NVS", "LIN", "MCD", "NFLX", "AZN", "ACN", "CRM", "BABA", "FMX", "CSCO", "SHEL", "BAC", "PEP", "ABBV", "KO", "TM", "COST", "MRK", "ASML", "CVX", "ADBE", "HD", "ORCL", "JNJ", "PG", "MA", "AVGO", "XOM", "JPM", "WMT", "NVO", "UNH", "TSM", "V", "LLY", "TSLA", "BRK-B", "META", "NVDA", "AMZN", "GOOGL", "MSFT", "AAPL"]

data = pd.read_csv('rubbish_data/AAPL.csv')

# Split the data into independent variables (features) and target variable
X = data.drop(['Close', 'Date'], axis=1)
y = data['Close']

# Split the data into training and testing subsets based on index
split_index = 1283
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and fit the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)
print(y_pred)
# Evaluate the model using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
mse_list.append(mse)
print("Mean Squared Error:", mse)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
plt.plot(range(len(y_test)), y_pred, color='r', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Price for AAPL')
plt.legend()
plt.show()

"""
for stock in stockList:
    data = pd.read_csv('Data/' + stock + '.csv')

    # Split the data into independent variables (features) and target variable
    X = data.drop(['Close', 'Date'], axis=1)
    y = data['Close']
    

    # Split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the multiple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)
    print("Mean Squared Error:", mse)
    """"""
    # Plot the graph of y_test and y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
    plt.plot(range(len(y_test)), y_pred, color='r', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Stock Price')
    plt.title('Actual vs. Predicted Stock Price for ' + stock)
    plt.legend()
    plt.show()
    

print(mse_list)
"""