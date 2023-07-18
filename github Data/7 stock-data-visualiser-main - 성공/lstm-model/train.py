# Stock Data Visualiser
# https://github.com/cytronicoder/stock-data-visualiser/tree/main
# 실행하면 학습한 train 결과랑 validation 결과 나오고 예측 값 plot으로 나옴

# Data analysis

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#####          여기 잘봐라 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# Custom modules
from config import config
from data import download_data, prepare_x, prepare_y
from helper import Normalizer, LSTMModel, TimeSeriesDataset

data_date, data_close_price, num_data_points, display_date_range = download_data(config)

# Normalize data
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

data_x, _data_x = prepare_x(
    normalized_data_close_price, window_size=config["data"]["window_size"]
)

data_y = prepare_y(
    normalized_data_close_price, window_size=config["data"]["window_size"]
)

# Split data into train and test
split_index = int(data_y.shape[0] * config["data"]["train_split_size"])

data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]

data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]


# Create dataset and dataloader for training
dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape:", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape:", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(
    dataset_train, batch_size=config["training"]["batch_size"], shuffle=True
)

val_dataloader = DataLoader(
    dataset_val, batch_size=config["training"]["batch_size"], shuffle=True
)

# Build model
model = LSTMModel(
    input_size=config["model"]["input_size"],
    hidden_layer_size=config["model"]["lstm_size"],
    num_layers=config["model"]["num_lstm_layers"],
    output_size=1,
    dropout=config["model"]["dropout"],
)

model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    betas=(0.9, 0.98),
    eps=1e-9,
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1
)

# Run an epoch
def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.detach().item() / batchsize

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print(
        "Epoch {}/{} | Loss - train: {:.6f} test: {:.6f} | lr: {:.6f}".format(
            epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train
        )
    )


# Re-initialize dataloader so data doesn't shuffle
# Plot the values by date
train_dataloader = DataLoader(
    dataset_train, batch_size=config["training"]["batch_size"], shuffle=False
)

val_dataloader = DataLoader(
    dataset_val, batch_size=config["training"]["batch_size"], shuffle=False
)

model.eval()

# Predict on the training data
predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# Predict on the validation data
predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

# Prepare data for plotting
y_train_pred = np.zeros(num_data_points)
y_val_pred = np.zeros(num_data_points)

y_train_pred[
    config["data"]["window_size"] : split_index + config["data"]["window_size"]
] = scaler.inverse_transform(predicted_train)

y_val_pred[split_index + config["data"]["window_size"] :] = scaler.inverse_transform(
    predicted_val
)

y_train_pred = np.where(y_train_pred == 0, None, y_train_pred)
y_val_pred = np.where(y_val_pred == 0, None, y_val_pred)

# Plot the values by date
fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))

plt.title(
    "Compare predicted prices to actual prices for {}".format(
        config["alpha_vantage"]["symbol"]
    )
)

xticks = [
    data_date[i]
    if (
        (
            i % config["plots"]["xticks_interval"] == 0
            and (num_data_points - i) > config["plots"]["xticks_interval"]
        )
        or i == num_data_points - 1
    )
    else None
    for i in range(num_data_points)
]

x = np.arange(0, len(xticks))
plt.xticks(x, xticks, rotation="vertical")
plt.grid(visible=True, which="major", axis="y", linestyle="--")

plt.ylabel("Price (USD)")

plt.plot(
    data_date,
    data_close_price,
    label="Actual prices",
    color=config["plots"]["color_actual"],
)

plt.plot(
    data_date,
    y_train_pred,
    label="Predicted prices (train)",
    color=config["plots"]["color_pred_train"],
)

plt.plot(
    data_date,
    y_val_pred,
    label="Predicted prices (validation)",
    color=config["plots"]["color_pred_val"],
)

plt.legend()
plt.show()

# Plotting the zoomed in view of the predicted prices vs. actual prices
y_val_subset = scaler.inverse_transform(data_y_val)
predicted_val = scaler.inverse_transform(predicted_val)
date = data_date[split_index + config["data"]["window_size"] :]

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))

plt.title("Zoom in to examine predicted price on validation data portion")

xticks = [
    date[i]
    if (
        (
            i % int(config["plots"]["xticks_interval"] / 5) == 0
            and (len(date) - i) > config["plots"]["xticks_interval"] / 6
        )
        or i == len(date) - 1
    )
    else None
    for i in range(len(date))
]

xs = np.arange(0, len(xticks))
plt.xticks(xs, xticks, rotation="vertical")
plt.grid(visible=True, which="major", axis="y", linestyle="--")

plt.ylabel("Price (USD)")

plt.plot(
    date,
    y_val_subset,
    label="Actual prices",
    color=config["plots"]["color_actual"],
)

plt.plot(
    date,
    predicted_val,
    label="Predicted prices (validation)",
    color=config["plots"]["color_pred_val"],
)

plt.legend()
plt.show()

# Predict the closing price of the next trading day
model.eval()

x = (
    torch.tensor(_data_x)
    .float()
    .to(config["training"]["device"])
    .unsqueeze(0)
    .unsqueeze(2)
)

prediction = model(x)
prediction = prediction.cpu().detach().numpy()

plot_range = 10
y_test_pred = np.zeros(plot_range)

y_test_pred[plot_range - 1] = scaler.inverse_transform(prediction)

y_test_pred = np.where(y_test_pred == 0, None, y_test_pred)

print(
    "Predicted close price of the next trading day: ${:.2f}".format(
        y_test_pred[plot_range - 1]
    )
)
