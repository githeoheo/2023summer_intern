"""
Configuration file from official Alpha Vantage API
Using the demo key for now - getting an academic key ASAP
"""

config = {
    "alpha_vantage": {
        "key": "demo",  # Demo key from Alpha Vantage to test the code
        "symbol": "IBM",  # IBM stock for demo purposes
        "output_size": "full",  # Get the full history of the stock
        "key_adjusted_close": "5. adjusted close",  # Key for adjusted close price
    },
    "data": {
        "window_size": 20,  # Number of days to use for prediction
        "train_split_size": 0.80,  # 80% of data for training, 20% for validation
    },
    "plots": {
        "xticks_interval": 90,  # Plot xticks every 90 days
        "color_actual": "#001f3f",  # Navy blue
        "color_train": "#3D9970",  # Green
        "color_val": "#0074D9",  # Blue
        "color_pred_train": "#3D9970",  # Green
        "color_pred_val": "#0074D9",  # Blue
        "color_pred_test": "#FF4136",  # Red
    },
    "model": {
        "input_size": 1,  # Number of features
        "num_lstm_layers": 2,  # Number of LSTM layers
        "lstm_size": 32,  # Number of units in each LSTM layer
        "dropout": 0.2,  # Dropout rate
    },
    "training": {
        "device": "cpu",  # Use "cuda" if you have a GPU
        
        # Adjust the following parameters to your needs
        "batch_size": 32,  # Number of samples per batch
        "num_epoch": 100,  # Number of epochs
        "learning_rate": 0.01,  # Learning rate
        "scheduler_step_size": 40,  # Step size for learning rate scheduler
    },
}
