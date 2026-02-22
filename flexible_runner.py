import os
import sys
import json
import datetime
from typing import Dict

import numpy as np
import pandas as pd
import keras as ks
import tensorflow as tf
from numpy.random import seed
from scipy import stats
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

seed(1)
tf.random.set_seed(1)
tf.config.optimizer.set_jit(True)


def parse_common_args(argv):
    if len(argv) < 13:
        raise ValueError(
            "Usage: script gw_file met_file_or_NONE output_dir lr seq_min seq_max dense_min dense_max batch_min batch_max units_min units_max"
        )

    met_file = argv[2].strip()
    if met_file.upper() in {"", "NONE", "NULL", "-"}:
        met_file = None

    return {
        "gw_file": argv[1],
        "met_file": met_file,
        "output_dir": argv[3],
        "learning_rate": float(argv[4]),
        "seq_min": int(argv[5]),
        "seq_max": int(argv[6]),
        "dense_min": int(argv[7]),
        "dense_max": int(argv[8]),
        "batch_min": int(argv[9]),
        "batch_max": int(argv[10]),
        "unit_min": int(argv[11]),
        "unit_max": int(argv[12]),
    }


def load_data(gw_path: str, met_path: str | None, use_shift: bool):
    gw = pd.read_csv(gw_path, parse_dates=["Date"], index_col=0, dayfirst=True, decimal=".", sep=",")
    gw = gw[["GWL"]].dropna().sort_index()

    data = gw.copy()
    used_meteo = False
    if met_path and os.path.exists(met_path):
        met = pd.read_csv(met_path, parse_dates=["Date"], index_col=0, dayfirst=True, decimal=".", sep=",")
        met = met.sort_index()
        data = pd.merge(gw, met, left_index=True, right_index=True, how="inner")
        used_meteo = True

    if use_shift:
        shift_col = gw[["GWL"]].copy()
        shift_col.index = shift_col.index.shift(1, freq="MS")
        shift_col.rename(columns={"GWL": "GWLt-1"}, inplace=True)
        data = pd.merge(data, shift_col, left_index=True, right_index=True, how="inner")

    well_id = os.path.splitext(os.path.basename(gw_path))[0]
    return data.dropna(), well_id, used_meteo


def split_data(data: pd.DataFrame, seq_length: int):
    n = len(data)
    train_end = round(0.60 * n)
    stop_end = round(0.75 * n)
    opt_end = round(0.80 * n)

    train = data.iloc[:train_end]
    stop = data.iloc[train_end:stop_end]
    stop_ext = data.iloc[max(0, train_end - seq_length):stop_end]
    opt = data.iloc[stop_end:opt_end]
    opt_ext = data.iloc[max(0, stop_end - seq_length):opt_end]
    test = data.iloc[opt_end:]
    test_ext = pd.concat([data.iloc[max(0, opt_end - seq_length):opt_end], test])
    return train, stop, stop_ext, opt, opt_ext, test, test_ext


def to_supervised(values: np.ndarray, seq_length: int):
    x, y = [], []
    for i in range(len(values) - seq_length):
        end = i + seq_length
        x.append(values[i:end, :])
        y.append(values[end, 0])
    return np.array(x), np.array(y)


def build_model(model_type: str, settings: Dict, x_train: np.ndarray):
    if model_type == "CNN":
        inputs = ks.layers.Input(shape=(settings["seq_length"], x_train.shape[2]))
        x = ks.layers.Conv1D(filters=settings["units"], kernel_size=3, padding="same", activation="relu")(inputs)
        x = ks.layers.MaxPooling1D(padding="same")(x)
        x = ks.layers.Flatten()(x)
        x = ks.layers.Dense(settings["dense_size"], activation="relu")(x)
        outputs = ks.layers.Dense(1, activation="linear")(x)
        model = ks.models.Model(inputs=inputs, outputs=outputs)
    else:
        model = ks.models.Sequential([
            ks.layers.Input(shape=(settings["seq_length"], x_train.shape[2])),
            ks.layers.LSTM(settings["units"], dropout=0.0, unit_forget_bias=True),
            ks.layers.Dense(1, activation="linear"),
        ])

    optimizer = ks.optimizers.Adam(learning_rate=settings["learning_rate"], epsilon=1e-3, clipnorm=1)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
    return model


def train_member(model_type, settings, x_train, y_train, x_stop, y_stop, ini):
    seed(ini)
    tf.random.set_seed(ini)
    model = build_model(model_type, settings, x_train)
    es = ks.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=0)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_stop, y_stop),
        epochs=40,
        batch_size=settings["batch_size"],
        verbose=0,
        callbacks=[es],
    )
    return model


def run_pipeline(model_type: str, use_shift: bool):
    cfg = parse_common_args(sys.argv)
    data, well_id, used_meteo = load_data(cfg["gw_file"], cfg["met_file"], use_shift)

    if len(data) < max(24, cfg["seq_max"] + 5):
        raise ValueError("داده کافی نیست. حداقل طول سری زمانی باید بیشتر باشد.")

    scaler = StandardScaler()
    scaler.fit(data)
    data_n = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    scaler_gwl = StandardScaler()
    scaler_gwl.fit(data[["GWL"]])

    def objective(densesize, seqlength, batchsize, units):
        settings = {
            "seq_length": int(seqlength),
            "dense_size": int(densesize),
            "batch_size": int(batchsize),
            "units": int(units),
            "learning_rate": cfg["learning_rate"],
        }

        train, _, stop_ext, _, opt_ext, _, _ = split_data(data, settings["seq_length"])
        train_n, _, stop_ext_n, _, opt_ext_n, _, _ = split_data(data_n, settings["seq_length"])

        x_train, y_train = to_supervised(train_n.values, settings["seq_length"])
        x_stop, y_stop = to_supervised(stop_ext_n.values, settings["seq_length"])
        x_opt, y_opt = to_supervised(opt_ext_n.values, settings["seq_length"])

        preds = np.zeros((len(x_opt), 3))
        for ini in range(3):
            model = train_member(model_type, settings, x_train, y_train, x_stop, y_stop, ini)
            pred_n = model.predict(x_opt, verbose=0)
            preds[:, ini] = scaler_gwl.inverse_transform(pred_n).ravel()

        sim = np.median(preds, axis=1)
        obs = scaler_gwl.inverse_transform(y_opt.reshape(-1, 1)).ravel()
        err = sim - obs
        err_nash = obs - np.mean(train["GWL"])
        r = stats.linregress(sim, obs)
        return (1 - (np.sum(err**2) / np.sum(err_nash**2))) + (r.rvalue**2)

    pbounds = {
        "seqlength": (cfg["seq_min"], cfg["seq_max"]),
        "densesize": (cfg["dense_min"], cfg["dense_max"]),
        "batchsize": (cfg["batch_min"], cfg["batch_max"]),
        "units": (cfg["unit_min"], cfg["unit_max"]),
    }

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1, verbose=0)
    well_dir = os.path.join(cfg["output_dir"], well_id)
    os.makedirs(well_dir, exist_ok=True)
    logger = JSONLogger(path=os.path.join(well_dir, f"logs_{model_type}_{well_id}.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    utility = UtilityFunction(kind="ei", kappa=0.05, xi=0.0)
    optimizer.maximize(init_points=8, n_iter=8, acquisition_function=utility)

    best = optimizer.max["params"]
    settings = {
        "seq_length": int(best["seqlength"]),
        "dense_size": int(best["densesize"]),
        "batch_size": int(best["batchsize"]),
        "units": int(best["units"]),
        "learning_rate": cfg["learning_rate"],
    }

    train_n, _, stop_ext_n, _, _, _, test_ext_n = split_data(data_n, settings["seq_length"])
    train, _, _, _, _, test, _ = split_data(data, settings["seq_length"])
    x_train, y_train = to_supervised(train_n.values, settings["seq_length"])
    x_stop, y_stop = to_supervised(stop_ext_n.values, settings["seq_length"])
    x_test, y_test = to_supervised(test_ext_n.values, settings["seq_length"])

    members = np.zeros((len(x_test), 5))
    models = []
    for ini in range(5):
        model = train_member(model_type, settings, x_train, y_train, x_stop, y_stop, ini)
        models.append(model)
        pred_n = model.predict(x_test, verbose=0)
        members[:, ini] = scaler_gwl.inverse_transform(pred_n).ravel()

    sim = np.median(members, axis=1)
    obs = scaler_gwl.inverse_transform(y_test.reshape(-1, 1)).ravel()
    err = sim - obs
    nse = 1 - (np.sum(err**2) / np.sum((obs - np.mean(train["GWL"])) ** 2))
    r2 = stats.linregress(sim, obs).rvalue**2
    rmse = float(np.sqrt(np.mean(err**2)))

    last_seq = data_n.iloc[-settings["seq_length"] :, :].values.reshape(1, settings["seq_length"], data_n.shape[1])
    next_scaled = np.median([m.predict(last_seq, verbose=0)[0, 0] for m in models])
    next_gwl = scaler_gwl.inverse_transform([[next_scaled]])[0, 0]
    next_date = data.index[-1] + pd.DateOffset(months=1)

    metadata = {
        "best_params": best,
        "learning_rate": cfg["learning_rate"],
        "model_type": model_type,
        "scenario": "GWLt-1" if use_shift else "GWL",
        "used_meteorological_data": used_meteo,
        "meteorological_file": cfg["met_file"] or "NONE",
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(os.path.join(well_dir, f"best_hyperparameters_{model_type}_{well_id}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(os.path.join(well_dir, f"log_summary_{model_type}_{well_id}.txt"), "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"Model: {model_type}",
                    f"Scenario: {'GWLt-1' if use_shift else 'GWL'}",
                    f"Used meteo data: {'Yes' if used_meteo else 'No'}",
                    f"NSE: {nse:.4f}",
                    f"R2: {r2:.4f}",
                    f"RMSE: {rmse:.4f}",
                    f"Forecast Date: {next_date.date()}",
                    f"Forecast GWL: {next_gwl:.3f}",
                ]
            )
        )

    plt.figure(figsize=(14, 6))
    plt.plot(test.index, obs, label="Observed")
    plt.plot(test.index, sim, label="Simulated")
    plt.scatter([next_date], [next_gwl], label="1-month forecast", color="red")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(well_dir, f"{well_id}_{model_type}_test_and_forecast.png"), dpi=300)
    plt.close()

    mode_desc = "با داده هواشناسی" if used_meteo else "فقط با سری تراز آب"
    print(f"✅ {model_type} finished for {well_id} ({mode_desc}) | NSE={nse:.3f}, R2={r2:.3f}, RMSE={rmse:.3f}")


if __name__ == "__main__":
    raise SystemExit("Use dedicated entry scripts.")
