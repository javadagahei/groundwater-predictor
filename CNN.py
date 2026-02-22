# ================================================
# CNN Univariate - فقط با داده تراز آب + پیش‌بینی ۱ ماه آینده
# ================================================

import sys
import os
import datetime
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
import pandas as pd
import keras as ks
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import mixed_precision

tf.config.optimizer.set_jit(True)
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

# برای تست سریع - بعداً حذف یا کامنت کن
if len(sys.argv) < 2:
    print("هیچ آرگومانی داده نشده → حالت تست فعال شد")
    sys.argv = [
        sys.argv[0],                              # نام اسکریپت
        r"d:\مقالات\بنیاد\Data\4303_GWdata.csv",  # gw_file
        "dummy",                                  # met_file (اگر لازم نیست)
        r"d:\results",                            # pathRes
        "0.001",                                   # lr
        "3", "12",                                # seq
        "16", "128",                              # dense
        "8", "64",                                # batch
        "16", "128"                               # filters
    ]



# ================== ورودی‌های خط فرمان (از GUI) ==================
gw_file   = sys.argv[1]      # مسیر فایل تراز آب
pathRes   = sys.argv[3]      # پوشه ذخیره نتایج
lr        = float(sys.argv[4])

seq_min   = int(sys.argv[5])
seq_max   = int(sys.argv[6])
dense_min = int(sys.argv[7])
dense_max = int(sys.argv[8])
batch_min = int(sys.argv[9])
batch_max = int(sys.argv[10])
filters_min = int(sys.argv[11])
filters_max = int(sys.argv[12])

# ================== توابع ==================
def load_gwl_data(gw_path):
    df = pd.read_csv(gw_path, parse_dates=['Date'], index_col=0, dayfirst=True, decimal='.', sep=',')
    data = df[['GWL']].copy().dropna()
    data = data.sort_index()
    Well_ID = os.path.splitext(os.path.basename(gw_path))[0]
    return data, Well_ID

def split_data(data, seq_length):
    n = len(data)
    train_end = round(0.60 * n)
    stop_end  = round(0.75 * n)
    opt_end   = round(0.80 * n)

    TrainingData = data.iloc[:train_end]
    StopData     = data.iloc[train_end:stop_end]
    StopData_ext = data.iloc[max(0, train_end - seq_length):stop_end]
    
    OptData      = data.iloc[stop_end:opt_end]
    OptData_ext  = data.iloc[max(0, stop_end - seq_length):opt_end]
    
    TestData     = data.iloc[opt_end:]
    TestData_ext = pd.concat([data.iloc[max(0, opt_end - seq_length):opt_end], TestData])

    return TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext

def to_supervised(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])      # شکل: (seq_length, 1)
        y.append(data[i+seq_length])        # فقط یک مقدار
    return np.array(X), np.array(y)

def gwmodel(ini, GLOBAL_SETTINGS, X_train, Y_train, X_stop, Y_stop):
    seed(ini)
    tf.random.set_seed(ini)
    
    inputs = ks.layers.Input(shape=(GLOBAL_SETTINGS["seq_length"], 1))   # univariate
    x = ks.layers.Conv1D(filters=GLOBAL_SETTINGS["filters"], kernel_size=3,
                         padding='same', activation='relu')(inputs)
    x = ks.layers.MaxPooling1D(padding='same')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu')(x)
    outputs = ks.layers.Dense(1, activation='linear')(x)
    
    model = ks.models.Model(inputs=inputs, outputs=outputs)
    optimizer = ks.optimizers.Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"],
                                   epsilon=1e-3, clipnorm=1)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    
    es = ks.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0)
    model.fit(X_train, Y_train, validation_data=(X_stop, Y_stop),
              epochs=30, batch_size=GLOBAL_SETTINGS["batch_size"],
              verbose=0, callbacks=[es])
    return model

# ================== تابع بهینه‌سازی بیزین ==================
def bayesOpt_function(densesize, seqlength, batchsize, filters):
    return bayesOpt_function_discrete(int(densesize), int(seqlength), int(batchsize), int(filters))

def bayesOpt_function_discrete(densesize_int, seqlength_int, batchsize_int, filters_int):
    GLOBAL_SETTINGS = {
        'batch_size': batchsize_int,
        'dense_size': densesize_int,
        'filters': filters_int,
        'seq_length': seqlength_int,
        'learning_rate': lr,
        'epochs': 60,
    }

    data, Well_ID = load_gwl_data(gw_file)
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, _, _ = split_data(data, seqlength_int)
    
    scaler_gwl = StandardScaler()
    scaler_gwl.fit(data[['GWL']])
    data_n = pd.DataFrame(scaler_gwl.transform(data), index=data.index, columns=['GWL'])
    
    Training_n, _, Stop_ext_n, Opt_n, Opt_ext_n, _, _ = split_data(data_n, seqlength_int)
    
    X_train, Y_train = to_supervised(Training_n.values, seqlength_int)
    X_stop,  Y_stop  = to_supervised(Stop_ext_n.values, seqlength_int)
    X_opt,   Y_opt   = to_supervised(Opt_ext_n.values,  seqlength_int)

    inimax = 5
    optresults = np.zeros((len(X_opt), inimax))
    for ini in range(inimax):
        model = gwmodel(ini, GLOBAL_SETTINGS, X_train, Y_train, X_stop, Y_stop)
        pred_n = model.predict(X_opt, verbose=0)
        pred = scaler_gwl.inverse_transform(pred_n)
        optresults[:, ini] = pred.ravel()

    sim = np.median(optresults, axis=1).reshape(-1, 1)
    obs = scaler_gwl.inverse_transform(Y_opt.reshape(-1, 1))
    err = sim - obs
    mean_train_stop = np.mean(np.concatenate([TrainingData['GWL'], StopData['GWL']]))
    err_nash = obs - mean_train_stop
    r = stats.linregress(sim.ravel(), obs.ravel())

    return (1 - (np.sum(err**2) / np.sum(err_nash**2))) + r.rvalue**2

# ================== ارزیابی تست + پیش‌بینی ۱ ماه آینده ==================
def run_evaluation_and_forecast(best_params):
    densesize_int = int(best_params['densesize'])
    seqlength_int = int(best_params['seqlength'])
    batchsize_int = int(best_params['batchsize'])
    filters_int   = int(best_params['filters'])

    GLOBAL_SETTINGS = {
        'batch_size': batchsize_int,
        'dense_size': densesize_int,
        'filters': filters_int,
        'seq_length': seqlength_int,
        'learning_rate': lr,
        'epochs': 60,
    }

    data, Well_ID = load_gwl_data(gw_file)
    scaler_gwl = StandardScaler()
    scaler_gwl.fit(data[['GWL']])
    data_n = pd.DataFrame(scaler_gwl.transform(data), index=data.index, columns=['GWL'])

    TrainingData, StopData, Stop_ext, OptData, Opt_ext, TestData, Test_ext = split_data(data, seqlength_int)
    Training_n, Stop_n, Stop_ext_n, Opt_n, Opt_ext_n, Test_n, Test_ext_n = split_data(data_n, seqlength_int)

    X_train, Y_train = to_supervised(Training_n.values, seqlength_int)
    X_stop,  Y_stop  = to_supervised(Stop_ext_n.values, seqlength_int)
    X_test,  Y_test  = to_supervised(Test_ext_n.values, seqlength_int)

    # انسامبل ۱۰ عضوی روی تست
    inimax = 10
    test_members = np.zeros((len(X_test), inimax))
    models = []
    for ini in range(inimax):
        model = gwmodel(ini, GLOBAL_SETTINGS, X_train, Y_train, X_stop, Y_stop)
        models.append(model)
        pred_n = model.predict(X_test, verbose=0)
        test_members[:, ini] = scaler_gwl.inverse_transform(pred_n).ravel()

    sim = np.median(test_members, axis=1)
    obs = scaler_gwl.inverse_transform(Y_test.reshape(-1, 1)).ravel()

    # محاسبه معیارها
    err = sim - obs
    err_nash = obs - np.mean(TrainingData['GWL'])
    NSE = 1 - (np.sum(err**2) / np.sum(err_nash**2))
    r = stats.linregress(sim, obs)
    R2 = r.rvalue**2
    RMSE = np.sqrt(np.mean(err**2))

    # ================== پیش‌بینی یک ماه آینده ==================
    last_seq = data_n.iloc[-seqlength_int:].values.reshape(1, seqlength_int, 1)   # آخرین پنجره
    forecast_scaled = np.median([m.predict(last_seq, verbose=0)[0,0] for m in models])
    forecast_gwl = scaler_gwl.inverse_transform([[forecast_scaled]])[0,0]

    next_date = data.index[-1] + pd.DateOffset(months=1)

    # ================== ذخیره و پلات ==================
    well_dir = os.path.join(pathRes, Well_ID)
    os.makedirs(well_dir, exist_ok=True)
    os.chdir(well_dir)
    
    import json
    with open(os.path.join(well_dir, f"best_hyperparameters_{Well_ID}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_params": {k: float(v) for k, v in best_params.items()},
            "learning_rate": lr,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Well_ID": Well_ID
        }, f, indent=4, ensure_ascii=False)

    # ۲. ذخیره لاگ خلاصه (نسخه کامل‌تر)
    with open(os.path.join(well_dir, f"log_summary_CNN_{Well_ID}.txt"), "w", encoding="utf-8") as f:
        f.write(f"""تاریخ اجرا: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Well_ID: {Well_ID}
بهترین پارامترها:
- sequence length : {seqlength_int}
- dense size      : {densesize_int}
- batch size      : {batchsize_int}
- filters         : {filters_int}
- learning rate   : {lr}

معیارهای عملکرد روی مجموعه تست:
NSE     : {NSE:.4f}
R²      : {R2:.4f}
RMSE    : {RMSE:.4f}

پیش‌بینی یک ماه آینده:
تاریخ پیش‌بینی     : {next_date.date()}
مقدار پیش‌بینی GWL : {forecast_gwl:.3f} متر بالای سطح دریا
""")
    # پلات تست + آینده
    plt.figure(figsize=(16, 7))
    plt.plot(TestData.index, obs, 'b-', label='Observed (Test)')
    plt.plot(TestData.index, sim, 'r-', label='Simulated (median)')
    plt.plot([TestData.index[-1], next_date], [obs[-1], forecast_gwl], 'r--', label='1-month forecast')
    plt.scatter(next_date, forecast_gwl, color='red', s=80, zorder=5)
    plt.title(f"CNN Univariate - {Well_ID} | 1-month forecast: {forecast_gwl:.3f} m")
    plt.ylabel('GWL [m asl]')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{Well_ID}_test_and_forecast.png", dpi=300)
    plt.show()
    plt.close()
    # لاگ
    with open(f'log_summary_CNN_{Well_ID}.txt', 'w', encoding='utf-8') as f:
        f.write(f"""NSE     {NSE:.3f}
R²      {R2:.3f}
RMSE    {RMSE:.3f}

Best params:

seq_length   = {seqlength_int}
dense_size   = {densesize_int}
batch_size   = {batchsize_int}
filters      = {filters_int}

1-month forecast = {forecast_gwl:.3f} m asl  (date: {next_date.date()})
""")

    print(f"\n✅ تمام! پیش‌بینی یک ماه آینده برای {Well_ID}: {forecast_gwl:.3f} متر")

# ================== اجرای اصلی ==================
if __name__ == "__main__":
    with tf.device("/cpu:0"):
        data, Well_ID = load_gwl_data(gw_file)
        well_dir = os.path.join(pathRes, Well_ID)
        os.makedirs(well_dir, exist_ok=True)
        os.chdir(well_dir)

        pbounds = {
            'seqlength': (seq_min, seq_max),
            'densesize': (dense_min, dense_max),
            'batchsize': (batch_min, batch_max),
            'filters':   (filters_min, filters_max),
        }

        optimizer = BayesianOptimization(
            f=bayesOpt_function,
            pbounds=pbounds,
            random_state=1,
            verbose=0
        )

        logger = JSONLogger(path=f"logs_CNN_{Well_ID}.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        utility = UtilityFunction(kind="ei", kappa=0.05, xi=0.0)
        optimizer.maximize(init_points=12, n_iter=0, acquisition_function=utility)

        # ادامه بهینه‌سازی تا ۵۰ تکرار یا عدم بهبود ۱۰ تایی
        while len(optimizer.res) < 50:
            best_step = max(range(len(optimizer.res)), key=lambda i: optimizer.res[i]['target'])
            if len(optimizer.res) - best_step > 10:
                break
            optimizer.maximize(init_points=0, n_iter=1, acquisition_function=utility)

        best_params = optimizer.max['params']
        print("\n Best parameter", best_params)

        # ارزیابی نهایی + پیش‌بینی یک ماه آینده
        run_evaluation_and_forecast(best_params)