import csv
import math
from datetime import date


def month_starts(start_year=2016, start_month=1, count=96):
    y, m = start_year, start_month
    for _ in range(count):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            m = 1
            y += 1


def write_gw(path):
    rows = []
    base = 1200.0
    trend = -0.015
    for i, d in enumerate(month_starts()):
        seasonal = 0.9 * math.sin(2 * math.pi * (i % 12) / 12.0)
        low_freq = 0.3 * math.sin(2 * math.pi * i / 36.0)
        gwl = base + trend * i + seasonal + low_freq
        rows.append((d.strftime("%d/%m/%Y"), round(gwl, 3)))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "GWL"])
        w.writerows(rows)


def write_meteo(path):
    rows = []
    for i, d in enumerate(month_starts()):
        rain = max(0.0, 35 + 28 * math.sin(2 * math.pi * ((i + 2) % 12) / 12.0))
        et = max(0.0, 62 + 24 * math.sin(2 * math.pi * ((i + 8) % 12) / 12.0))
        temp = 19 + 10 * math.sin(2 * math.pi * ((i + 7) % 12) / 12.0)
        tsin = math.sin(2 * math.pi * (i % 12) / 12.0)
        rows.append((d.strftime("%d/%m/%Y"), round(rain, 3), round(et, 3), round(temp, 3), round(tsin, 6)))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Rain", "ET", "T", "Tsin"])
        w.writerows(rows)


def main():
    write_gw("sample_data/sample_gwl.csv")
    write_meteo("sample_data/sample_meteo.csv")
    print("Sample CSV files generated in sample_data/")


if __name__ == "__main__":
    main()
