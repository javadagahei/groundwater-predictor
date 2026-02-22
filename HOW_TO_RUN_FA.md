# راهنمای سریع استفاده از نرم‌افزار (با داده نمونه)

این پروژه الان ۴ حالت اجرا دارد:
- CNN + GWL
- CNN + GWLt-1
- LSTM + GWL
- LSTM + GWLt-1

فایل هواشناسی **اختیاری** است. اگر ندهید، مدل فقط با `GWL` پیش‌بینی یک‌ماهه انجام می‌دهد.

## 1) ساخت داده نمونه

```bash
python tools/create_sample_data.py
```

بعد از اجرا این دو فایل ساخته می‌شوند:
- `sample_data/sample_gwl.csv`
- `sample_data/sample_meteo.csv`

## 2) اجرای خط فرمان (بدون هواشناسی)

```bash
python CNN_seq2val.py \
  sample_data/sample_gwl.csv \
  NONE \
  sample_data/out_no_meteo \
  0.001 \
  3 8 \
  8 32 \
  8 32 \
  8 32
```

## 3) اجرای خط فرمان (با هواشناسی)

```bash
python LSTM_seq2val_GWLshift.py \
  sample_data/sample_gwl.csv \
  sample_data/sample_meteo.csv \
  sample_data/out_with_meteo \
  0.001 \
  3 8 \
  8 32 \
  8 32 \
  8 32
```

## 4) اجرای GUI

```bash
python gui_main.py.py
```

در GUI:
1. مدل (CNN/LSTM) را انتخاب کنید.
2. سناریو (GWL یا GWLt-1) را انتخاب کنید.
3. فایل تراز آب را بدهید (`sample_gwl.csv`).
4. فایل هواشناسی را اگر دارید بدهید؛ اگر ندارید خالی بگذارید.
5. پوشه خروجی را انتخاب کنید.
6. هایپرپارامترها را به صورت بازه وارد کنید.
7. اجرای پیش‌بینی را بزنید.

## خروجی‌ها

در پوشه خروجی (داخل فولدر نام چاه) این فایل‌ها ذخیره می‌شوند:
- `best_hyperparameters_<MODEL>_<WELL>.json`
- `log_summary_<MODEL>_<WELL>.txt`
- `logs_<MODEL>_<WELL>.json`
- `<WELL>_<MODEL>_test_and_forecast.png`

این ساختار برای استفاده کارشناسی (با/بدون داده هواشناسی) آماده است.
