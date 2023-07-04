# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
PATH = "../datasets/raw"
CRYPTO = "BTCUSDT"
# %%
dataset = pd.read_csv(f"{PATH}/4h/{CRYPTO}.csv", delimiter="|")
# %%
dataset


def dataset_transforamtion(dataset, lags=14, forecast=2):
    lr = dataset[["lr"]]
    for i in range(1, lags):
        lr[f"t-{i}"] = lr["lr"].shift(i)
    lr = lr.dropna().iloc[:, ::-1]
    return lr.iloc[:, :-forecast], lr.iloc[:, -forecast:]


X, Y = dataset_transforamtion(dataset, lags=35, forecast=2)
SS = StandardScaler()
X = SS.fit_transform(X)

# # %%
# def transformation(dataset, lags=10, pct=False):
#     if(pct):
#         lr = dataset[["lr"]].pct_change()
#     else:
#         lr = SS.fit_transform(dataset[["lr"]])
#         lr = pd.DataFrame(data=lr,columns=["lr"])
#     for i in range(1, lags):
#         lr[f"t-{i}"] = lr["lr"].shift(i)
#     return lr.dropna()

# def data_transformation(dataset, lags=10):
#     pct = dataset[["lr"]].pct_change()
#     for i in range(1, 2):
#         pct[f"t-{i}"] = pct["lr"].shift(i)
#     target = pct.prod(axis=1).apply(lambda x: 0 if x > 0 else 1)

#     lr = SS.fit_transform(dataset[["lr"]])
#     lr = pd.DataFrame(data=lr,columns=["lr"])
#     for i in range(1, lags):
#         lr[f"t-{i}"] = lr["lr"].shift(i)
#     # return pd.concat([target,lr.iloc[:,2:]],axis=1).iloc[:, ::-1].dropna()
#     return lr.iloc[lags:,2:] ,target[lags:]
# # %%
# tmp = transformation(dataset,lags=14, pct=False)
# tmp2 = transformation(dataset,lags=14, pct=True)
# # %%
# tmp

# # %%
# def enroque_detector(dataset, length=1, pct=True):
#     aux_col = dataset.columns[:length]
#     if pct:
#         lr = dataset[aux_col].pct_change().prod(axis=1).apply(lambda x: 0 if x > 0 else 1)
#     else:
#         aux_df = dataset[aux_col].prod(axis=1).apply(lambda x: 0 if x > 0 else 1)
#         dataset = dataset.drop(columns=aux_col)
#         dataset = dataset.iloc[:, ::-1]

#     return dataset, aux_df

# X, Y = data_transformation(dataset,lags=10)
# oversampler = RandomUnderSampler()
# X_resampled, y_resampled = oversampler.fit_resample(X, Y)

# %%
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, GaussianNoise, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


optimizer = Adam(learning_rate=0.0005)
# Create the LSTM model
model = Sequential()
# model.add(
#     GRU(512, return_sequences=True, input_shape=(12, 1))
# )  # Input shape is (timesteps, features)
# model.add(GaussianNoise(0.1))  # Input shape is (timesteps, features)
# model.add(GRU(256, return_sequences=False))  # Input shape is (timesteps, features)
# model.add(GaussianNoise(0.01))  # Input shape is (timesteps, features)
# model.add(Dense(128,activation="tanh"))  # Input shape is (timesteps, features)
# model.add(Dense(2))

# model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
# model.add(Dense(2))

model.add(
    LSTM(
        256,
        return_sequences=True,
        input_shape=(X.shape[1], 1),
        dropout=0.2,
        recurrent_dropout=0.2,
    )
)
model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(2))

# Compile the model
model.compile(loss="mean_squared_error", optimizer=optimizer)
early_stopping = EarlyStopping(monitor="val_loss", patience=10)

# Train the model
history = model.fit(
    X,
    Y,
    shuffle=True,
    callbacks=[early_stopping],
    epochs=20,
    batch_size=64,
    validation_split=0.2,
)
# %%
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()
# %%
X2 = SS.inverse_transform(X)
Y2 = model.predict(X)
# # %%
# X2[0]
# %%
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 10))

for ax in axes.flatten():
    index = np.random.randint(0, len(X2) + 1)
    data = np.concatenate((X2[index], Y.values[index]), axis=0)
    data2 = np.concatenate((np.zeros_like(X2[index]), Y2[index]), axis=0)

    # Compute the slopes
    slopes = np.diff(data)

    # Determine the colors based on slope sign
    colors = ["red" if slope < 0 else "green" for slope in slopes]

    # Create the bar plot
    ax.bar(range(len(data)), data, color=colors)
    ax.bar(range(len(data2)), data2, color="black", alpha=0.7)

    # Set the x-axis at 0
    ax.axvline(0, color="black")

    # Set the y-axis labels
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(range(len(data)))

plt.tight_layout()
plt.show()
# %%
for index in np.random.randint(0, len(X2) + 1, size=10):
    data = np.concatenate((X2[index], Y.values[index]), axis=0)
    data2 = np.concatenate((np.zeros_like(X2[index]), Y2[index]), axis=0)

    # Compute the slopes
    slopes = np.diff(data)

    # Determine the colors based on slope sign
    colors = ["red" if slope < 0 else "green" for slope in slopes]
    # Create the bar plot
    plt.figure(int(index))
    print(index)
    plt.bar(range(len(data)), data, color=colors)
    plt.bar(range(len(data2)), data2, color="black", alpha=0.7)

    # Set the x-axis at 0
    plt.axvline(0, color="black")

    # Set the y-axis labels
    plt.yticks(range(len(data)), range(len(data)))

    # # Set the ticks and labels
    # ax.set_yticks(range(len(data)))
    # ax.set_yticklabels(range(len(data), 0, -1))
    # ax.set_xlabel('Values')

    # Show the plot
    plt.show()
# %%
index = 6081
data = np.concatenate((X2[index], Y.values[index]), axis=0)
data2 = np.concatenate((np.zeros_like(X2[index]), Y2[index]), axis=0)

# Compute the slopes
slopes = np.diff(data)

# Determine the colors based on slope sign
colors = ["red" if slope < 0 else "green" for slope in slopes]
# Create the bar plot
plt.figure(int(index))
print(index)
plt.bar(range(len(data)), data, color=colors)
plt.bar(range(len(data2)), data2, color="black", alpha=0.7)

# Set the x-axis at 0
plt.axvline(0, color="black")

# Set the y-axis labels
plt.yticks(range(len(data)), range(len(data)))

# # Set the ticks and labels
# ax.set_yticks(range(len(data)))
# ax.set_yticklabels(range(len(data), 0, -1))
# ax.set_xlabel('Values')

# Show the plot
plt.show()
# %%

import pandas as pd
import plotly.graph_objects as go
import pandas_ta as ta
from plotly.subplots import make_subplots

# Assuming you have the necessary OHLC data in a DataFrame called 'data'
# and the linear regression oscillator (LRO) is calculated using ta library
data = dataset.iloc[-30:]
# Calculate the linear regression oscillator (LRO)
data["lro"] = data[["lr"]]
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("OHLC", "Volume"),
    row_width=[0.3, 0.7],
)

# candlestick
fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick",
    ),
    row=1,
    col=1,
)
# volume
slopes = data["lro"].diff()
colors01=['green' if slope > 0 else 'red' for slope in slopes]

fig.add_trace(
    go.Bar(x=data.index, y=data["lro"], name="LRO", marker_color=colors01),
    row=2,
    col=1,
)

fig.update_layout(
    title="BTC-PERP: 5-minute OHLCV", yaxis_title="Price (USD)", width=900, height=600
)

# hide the slide bar
fig.update(layout_xaxis_rangeslider_visible=False)
fig.show()
# %%
