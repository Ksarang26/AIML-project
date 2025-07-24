import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictor:
    def __init__(self, symbol='AAPL', period='2y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def fetch_data(self):
        """Fetch stock data - with backup sample data generation"""
        print(f"Fetching data for {self.symbol}...")
        
        try:
            # Try to import and use yfinance
            import yfinance as yf
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            self.data.reset_index(inplace=True)
            print(f"Real data fetched successfully! Shape: {self.data.shape}")
            
        except ImportError:
            print("yfinance not available. Generating sample data for demonstration...")
            self.generate_sample_data()
            
        except Exception as e:
            print(f"Error fetching real data: {e}")
            print("Generating sample data for demonstration...")
            self.generate_sample_data()
            
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        return self.data
    
    def generate_sample_data(self):
        """Generate realistic sample stock data for demonstration"""
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)  # About 2 years of data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # Remove weekends
        
        # Generate realistic stock price data
        np.random.seed(42)  # For reproducible results
        n_days = len(dates)
        
        # Starting price
        initial_price = 150.0
        
        # Generate price movements with trend and volatility
        daily_returns = np.random.normal(0.001, 0.02, n_days)  # Small upward trend with volatility
        daily_returns[0] = 0  # First day has no return
        
        # Add some market events (larger movements)
        event_days = np.random.choice(n_days, size=20, replace=False)
        daily_returns[event_days] += np.random.normal(0, 0.05, 20)
        
        # Calculate cumulative prices
        price_multipliers = np.cumprod(1 + daily_returns)
        close_prices = initial_price * price_multipliers
        
        # Generate OHLV data based on close prices
        opens = []
        highs = []
        lows = []
        volumes = []
        
        for i, close in enumerate(close_prices):
            # Open price (based on previous close with some gap)
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, 0.005)  # Small overnight gap
                open_price = close_prices[i-1] * (1 + gap)
            
            # High and low prices
            intraday_volatility = np.random.uniform(0.01, 0.03)
            high_price = max(open_price, close) * (1 + intraday_volatility * np.random.uniform(0.3, 1.0))
            low_price = min(open_price, close) * (1 - intraday_volatility * np.random.uniform(0.3, 1.0))
            
            # Volume (with some correlation to price movements)
            price_change = abs(close - open_price) / open_price
            base_volume = 50000000  # 50M base volume
            volume = int(base_volume * (1 + price_change * 5) * np.random.uniform(0.5, 2.0))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            volumes.append(volume)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes
        })
        
        print(f"Sample data generated successfully! Shape: {self.data.shape}")
        print("Note: This is simulated data for demonstration purposes.")
    
    # Rest of the methods remain the same as in the original code
    def create_technical_indicators(self):
        """Create technical indicators as features"""
        print("Creating technical indicators...")
        
        df = self.data.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Moving Average Ratios
        df['MA_5_20_ratio'] = df['MA_5'] / df['MA_20']
        df['MA_10_50_ratio'] = df['MA_10'] / df['MA_50']
        
        # Price-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2d'] = df['Close'].pct_change(periods=2)
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()
        
        # Lag features
        for i in [1, 2, 3, 5]:
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        self.data = df
        print("Technical indicators created successfully!")
        return df
    
    def prepare_features(self):
        """Prepare features and target variables"""
        print("Preparing features and target...")
        
        # Select feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'MA_5_20_ratio', 'MA_10_50_ratio',
            'High_Low_Pct', 'Open_Close_Pct',
            'Volume_Ratio', 'RSI', 'BB_Position',
            'Price_Change', 'Price_Change_2d', 'Price_Change_5d',
            'Volatility',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_5'
        ]
        
        # Create target variable (next day's closing price)
        self.data['Target'] = self.data['Close'].shift(-1)
        
        # Remove rows with NaN values
        clean_data = self.data.dropna()
        
        # Extract features and target
        self.features = clean_data[feature_columns]
        self.target = clean_data['Target']
        
        print(f"Features shape: {self.features.shape}")
        print(f"Target shape: {self.target.shape}")
        
        # Feature correlation analysis
        feature_corr = self.features.corrwith(self.target).sort_values(ascending=False)
        print("\nTop 10 features correlated with target:")
        print(feature_corr.head(10))
        
        return self.features, self.target
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        print("Splitting data...")
        
        # For time series, we don't shuffle the data
        split_index = int(len(self.features) * (1 - test_size))
        
        X_train = self.features.iloc[:split_index]
        X_test = self.features.iloc[split_index:]
        y_train = self.target.iloc[:split_index]
        y_test = self.target.iloc[split_index:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Testing set size: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models"""
        print("Training models...")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_pred': train_pred,
                'test_pred': test_pred
            }
            
            print(f"Train RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
    
    def evaluate_models(self, y_train, y_test):
        """Evaluate and compare model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Create comparison DataFrame
        metrics_data = []
        for name, results in self.results.items():
            metrics_data.append({
                'Model': name,
                'Train RMSE': results['train_rmse'],
                'Test RMSE': results['test_rmse'],
                'Train MAE': results['train_mae'],
                'Test MAE': results['test_mae'],
                'Train R²': results['train_r2'],
                'Test R²': results['test_r2']
            })
        
        comparison_df = pd.DataFrame(metrics_data)
        print(comparison_df.round(4))
        
        # Find best model
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['test_rmse'])
        print(f"\nBest performing model: {best_model}")
        
        return comparison_df
    
    def plot_predictions(self, y_train, y_test, days_to_show=100):
        """Plot actual vs predicted prices"""
        print("Plotting predictions...")
        
        # Get the best model
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['test_rmse'])
        
        # Get predictions
        train_pred = self.results[best_model]['train_pred']
        test_pred = self.results[best_model]['test_pred']
        
        # Create time indices
        total_days = len(y_train) + len(y_test)
        train_days = len(y_train)
        
        # Show only last part of the data for clarity
        start_idx = max(0, train_days - days_to_show)
        
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        train_range = range(start_idx, train_days)
        if start_idx < train_days:
            plt.plot(train_range, y_train.iloc[start_idx:], 
                    label='Actual (Train)', color='blue', linewidth=2)
            plt.plot(train_range, train_pred[start_idx:], 
                    label='Predicted (Train)', color='lightblue', linewidth=2, alpha=0.7)
        
        # Plot testing data
        test_range = range(train_days, total_days)
        plt.plot(test_range, y_test, 
                label='Actual (Test)', color='red', linewidth=2)
        plt.plot(test_range, test_pred, 
                label='Predicted (Test)', color='orange', linewidth=2, alpha=0.8)
        
        # Add vertical line to separate train/test
        plt.axvline(x=train_days, color='gray', linestyle='--', alpha=0.7, 
                   label='Train/Test Split')
        
        plt.title(f'{self.symbol} Stock Price Prediction - {best_model}', fontsize=16, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_quick_demo(self):
        """Run a quick demonstration for interview"""
        print("="*60)
        print(f"QUICK STOCK PREDICTION DEMO FOR {self.symbol}")
        print("="*60)
        
        # Fetch data
        self.fetch_data()
        
        # Create features
        self.create_technical_indicators()
        self.prepare_features()
        
        # Split and train
        X_train, X_test, y_train, y_test = self.split_data()
        self.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate
        results = self.evaluate_models(y_train, y_test)
        
        # Plot
        self.plot_predictions(y_train, y_test, days_to_show=50)
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        
        return results

# Run the demo
if __name__ == "__main__":
    # Create predictor instance
    predictor = StockPricePredictor(symbol='AAPL', period='1y')
    
    # Run quick demo
    results = predictor.run_quick_demo()
    
    print("\nThis demo shows:")
    print("✅ Data preprocessing and feature engineering")
    print("✅ Machine learning model training")
    print("✅ Model evaluation and comparison")
    print("✅ Visualization of predictions")
    print("\nProject is ready for interview demonstration!")