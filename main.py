import numpy as np
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import pandas as pd
import random
import datetime

# Solver'ın oluşturulması
solver = pywraplp.Solver.CreateSolver("SCIP")
if not solver:
    raise ValueError("SCIP çözücüsü bulunamadı!")

# kıdem aralıklarının çekilmesi
df = pd.read_excel("asistan_kidem.xlsx")
kıdem_aralıkları = []
for i in set(df["Kıdem"]):
    aralık = min(df[df["Kıdem"] == i].index + 1), max(df[df["Kıdem"] == i].index + 1)
    kıdem_aralıkları.append(aralık)
kıdem_aralıkları = kıdem_aralıkları[::-1]
df["Ad-Soyad"] = df.apply(lambda row: row["Ad"] + " " + row["Soyad"], axis=1)


year = 2024
month = 5  # her ay için güncellenmesi gerekiyor Nisan = 4, Mayıs = 5, Haziran = 6.... şeklinde
first_day_of_month = datetime.date(2024, month, 1).weekday()
n = calendar.monthrange(year, month)[1]
D = len(df)

# Lecture day's deviations
dev = {}
for i in range(1, D + 1):
    for j in range(1, n + 1):
        dev[i, j] = solver.IntVar(0, 1, f"dev[{i}][{j}]")
deveks = {}
for i in range(1, D + 1):
    for j in range(1, n + 1):
        deveks[i, j] = solver.IntVar(0, 1, f"deveks[{i}][{j}]")

# Sapma değerleri için değişkenlerin tanımlanması ilk 3 kıdem için için
dev1 = {}
for i in range(1, D + 1):
    for j in range(1, n + 1):
        dev1[i, j] = solver.IntVar(0, 1, f"dev1[{i}][{j}]")
deveks1 = {}
for i in range(1, D + 1):
    for j in range(1, n + 1):
        deveks1[i, j] = solver.IntVar(0, 1, f"deveks1[{i}][{j}]")


# Değişkenlerin tanımlanması
x = {}
for i in range(1, D + 1):
    for j in range(1, n + 1):
        x[i, j] = solver.BoolVar(f"x[{i}][{j}]")


# Her gün her kıdemden en az bir doktor
for j in range(1, n + 1):
    for aralık in kıdem_aralıkları:
        start, end = aralık
        if (end - start) <= 2:
            solver.Add(
                sum(x[i, j] for i in range(start, end + 1)) <= 1
            )  # eğer kıdemde yeterli kişi yoksa o kıdemden nöbet tutan kimse olmayabilir(örneğin kıdemdeki kişi sayısı 3 kişi ve altındaysa)
        else:
            solver.Add(
                1 <= sum(x[i, j] for i in range(start, end + 1))
            )  # o gün için o kıdemden çalışacak bütün doktorların toplamının sayısının 1 den büyük olması


# 6. kıdemi 4ten küçük yapma
for j in range(1, n + 1):
    start, end = (23, 33)
    solver.Add(sum(x[i, j] for i in range(start, end + 1)) <= 3)
    solver.Add(sum(x[i, j] for i in range(start, end + 1)) >= 2)
"""
#7. kıdem
for j in range(1, n+1):
  start, end = (34,36)
  solver.Add(sum(x[i, j] for i in range(start, end+1)) <= 2)
  solver.Add(sum(x[i, j] for i in range(start, end+1)) >= 1)
"""
# 4. kıdemi 3ten küçük yapma
for j in range(1, n + 1):
    start, end = (13, 17)
    solver.Add(sum(x[i, j] for i in range(start, end + 1)) <= 2)
"""
#5. kıdemi 3ten küçük yapma
for j in range(1, n+1):
  start, end = (18, 22)
  solver.Add(sum(x[i, j] for i in range(start, end+1)) <= 3)
"""

# iki günde bir nöbet -bir nöbetten  sonra iki gün boş kalmalı-
for i in range(1, D + 1):
    for j in range(1, n - 1):
        solver.Add(x[i, j] + x[i, j + 1] + x[i, j + 2] <= 1)


# bir önceki ayın son günü (last_day) ve sondan bir önceki günü (second_last_day) nöbetten sonra 2 gün dinlenme kısıtı
worked_last_day = {1, 6, 7, 8, 9, 15, 19, 28, 32}
worked_second_last_day = {2, 5, 12, 16, 17, 18, 27, 31}

# Constraints based on the last two days of the previous month
for i in range(1, D + 1):
    if i in worked_last_day:
        # Doctor cannot work on the 1st and 2nd days of the current month
        solver.Add(x[i, 1] == 0)
        solver.Add(x[i, 2] == 0)
    elif i in worked_second_last_day:
        # Doctor cannot work on the 1st day of the current month
        solver.Add(x[i, 1] == 0)


# Bir gündeki doktor sayısının olabildiğince eşit olması  (ortalamadan -1 olması kısıtı) bir günde 9 veya 10 doktor bulunması gibi

for j in range(1, n + 1):
    solver.Add(sum(x[i, j] for i in range(1, D + 1)) <= round(D * 8 / n))
    solver.Add(round(D * 8 / n) - 1 <= sum(x[i, j] for i in range(1, D + 1)))


# Bir gündeki doktor sayısının olabildiğince eşit olması  (ortalamadan  +1 olması kısıtı)

# for j in range(1, n+1):
#     solver.Add(sum(x[i, j] for i in range(1, D+1)) <= round(D*8/n)+1)
#     solver.Add(round(D*8/n) <= sum(x[i, j] for i in range(1, D+1)))


# Hafta içi ve hafta sonu nöbetlerinin eşit ya da olabildiğince eşit dağılması
weekdays = [0, 1, 2, 3]  # Pazartesi, Salı, Çarşamba, Perşembe
weekends = [4, 5, 6]  # Cuma, Cumartesi, Pazar

for i in range(1, D + 1):
    weekday_shifts = sum(
        x[i, j] for j in range(1, n + 1) if (j + first_day_of_month - 1) % 7 in weekdays
    )
    weekend_shifts = sum(
        x[i, j] for j in range(1, n + 1) if (j + first_day_of_month - 1) % 7 in weekends
    )
    solver.Add(weekday_shifts - weekend_shifts <= 2)
    solver.Add(-2 <= weekday_shifts - weekend_shifts)


salı_günleri = [
    j for j in range(1, n + 1) if (j + first_day_of_month - 1) % 7 == 1
]  # Salı
perşembe_günleri = [
    j for j in range(1, n + 1) if (j + first_day_of_month - 1) % 7 == 3
]  # Perşembe

# Her doktorun toplam Salı ve Perşembe günleri tuttuğu nöbet sayılarının olabildiğince eşit olması kısıtı
for i in range(1, D + 1):
    salı_perşembe_toplam_i = sum(x[i, gün] for gün in salı_günleri) + sum(
        x[i, gün] for gün in perşembe_günleri
    )
    deviation = sum(dev[i, gün] for gün in (salı_günleri + perşembe_günleri)) - sum(
        deveks[i, gün] for gün in (salı_günleri + perşembe_günleri)
    )
    solver.Add(salı_perşembe_toplam_i + deviation == 2)

# her doktor 8 nöbet tutmalı
for i in range(1, D + 1):
    # if i == 24:
    #   solver.Add(sum(x[i, j] for j in range(1, n+1)) == 2) #özel durum (24. doktor 2 nöbet tutacak)
    if i == 23:
        solver.Add(sum(x[i, j] for j in range(1, n + 1)) == 0)
    elif i == 5:
        solver.Add(sum(x[i, j] for j in range(1, n + 1)) == 7)
    else:
        solver.Add(sum(x[i, j] for j in range(1, n + 1)) == 8)


# doctor_availabilities -doktorların müsait oldukları, özellikle nöbet tutmak istedikleri günler
doctor_availabilities = {
    2: [],
    3: [],
    7: [],
    8: [],
    9: [],
    11: [],
    13: [],
    21: [],
    24: [],
    35: [],
}


# each doctor works on their available dates
for doctor, available_dates in doctor_availabilities.items():
    for date in available_dates:
        solver.Add(x[doctor, date] == 1)  # Doctor must work on their available dates


# doctor_non_availabilities -nöbet istenmeyen günler-
doctor_non_availabilities = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
    12: [],
    13: [],
    14: [],
    15: [],
    16: [],
    17: [],
    18: [],
    19: [],
    20: [],
    21: [],
    22: [],
    23: [],
    24: [],
    25: [],
    26: [],
    27: [],
    28: [],
    29: [],
    30: [],
    31: [],
    32: [],
    33: [],
    34: [],
    35: [],
    36: [],
}

# each doctor doesn't work on their unavailable dates
for doctor, unavailable_dates in doctor_non_availabilities.items():
    for date in unavailable_dates:
        solver.Add(
            x[doctor, date] == 0
        )  # Doctor cannot work on their unavailable dates


# Hedef fonksiyonu tanımlama
objective_terms = []
for doktor in range(1, D + 1):
    for gün in salı_günleri + perşembe_günleri:
        objective_terms.append(dev[doktor, gün])
        objective_terms.append(deveks[doktor, gün])

solver.Minimize(solver.Sum(objective_terms))

evli_çiftler = [(1, 6), (2, 11), (16, 17)]  # Çiftlerin doktor numaraları

# Her çift için, ay boyunca 6 gün birlikte ve 2 gün ayrı nöbet tutma kısıtları
for doktor1, doktor2 in evli_çiftler:
    # Her gün için birlikte nöbet tutup tutmadıklarını belirten değişkenler
    birlikte_nöbet = [
        solver.BoolVar(f"birlikte_nöbet[{doktor1}][{doktor2}][{j}]")
        for j in range(1, n + 1)
    ]

    for j in range(1, n + 1):
        # Eğer her iki doktor da aynı gün nöbetçi ise, birlikte_nöbet[j] 1 olur
        solver.Add(birlikte_nöbet[j - 1] <= x[doktor1, j])
        solver.Add(birlikte_nöbet[j - 1] <= x[doktor2, j])
        solver.Add(birlikte_nöbet[j - 1] >= x[doktor1, j] + x[doktor2, j] - 1)

    # Her çift için toplam 6 gün birlikte nöbet tutma kısıtı
    solver.Add(sum(birlikte_nöbet) == 6)


# Problem çözülmesi
status = solver.Solve()

# Çözümün yazdırılması
if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:

    for i in range(1, D + 1):
        for j in range(1, n + 1):
            if x[i, j].solution_value() > 0.5:
                print(df["Ad-Soyad"].iloc[i - 1] + f", Gün {j} için nöbettedir.")
else:
    print("Optimal çözüm bulunamadı!")

from collections import defaultdict
import pandas as pd

# Boş bir DataFrame
df1 = pd.DataFrame()

# Ayın günleri
days = [f"{day:02d}.4.2024" for day in range(1, 32)]  # aydan aya değişmeli
df1["Days"] = days


doktor_nobetleri = defaultdict(list)

for i in range(1, D + 1):
    # calisma = []
    # for j in range(1, n + 1):
    #     if x[i, j].solution_value() > 0.5:
    #         calisma.append(j)
    # doktor_calisma_gunleri.append(calisma)

    for j in range(1, n + 1):
        if x[i, j].solution_value() > 0.5:
            doktor_nobetleri[i].append(j)

# Doktor isimlerini, çalışma günleri ve kıdem aralıklarına göre ekleme
for doktor_id, nobet_gunleri in doktor_nobetleri.items():
    doktor_adi = df["Ad-Soyad"].iloc[doktor_id - 1]
    for gun in nobet_gunleri:
        for j, kıdem in enumerate(kıdem_aralıkları):
            if kıdem[0] <= doktor_id <= kıdem[1]:
                col_name_base = f"Kıdem {j + 1}"
                col_name = col_name_base
                if col_name not in df1:
                    df1[col_name] = ""
                while df1.at[gun - 1, col_name] != "":
                    j += 1
                    col_name = f"{col_name_base}_{j:02d}"
                    if col_name not in df1:
                        df1[col_name] = ""
                df1.at[gun - 1, col_name] = doktor_adi

# 'Days' sütununu 1. sütuna taşıma
days_column = df1.pop("Days")
df1.insert(0, "Days", days_column)

# Sütun başlıklarını numaralandırma
column_names = list(df1.columns)
prev_base = ""
prev_num = 0
for i in range(1, len(column_names)):
    if "_" in column_names[i]:
        base, num = column_names[i].split("_")
        if prev_base == base:
            prev_num += 1
        else:
            prev_base = base
            prev_num = 1
        column_names[i] = f"{base}_{prev_num}"
df1.columns = column_names


# DataFrame'i Excel dosyasına kaydetme
excel_file_path = "doktor_nobetleri.xlsx"
df1.to_excel(excel_file_path, index=False)

print(f"Data saved to '{excel_file_path}'")

import matplotlib.pyplot as plt
import numpy as np


def plot_schedule():
    fig, ax = plt.subplots(figsize=(15, 10))

    # Doktor isimleri ve kıdem bilgileri
    doktor_isimleri = df["Ad-Soyad"]
    seniority_levels = df["Kıdem"]

    doktor_calisma_gunleri = []
    for i in range(1, D + 1):
        calisma = []
        for j in range(1, n + 1):
            if x[i, j].solution_value() > 0.5:
                calisma.append(j)
        doktor_calisma_gunleri.append(calisma)

    cmap = plt.get_cmap(
        "viridis", len(kıdem_aralıkları) + 1
    )  # colormap       # kıdem aralıkları
    for idx, (isim, gunler) in enumerate(zip(doktor_isimleri, doktor_calisma_gunleri)):
        seniority = seniority_levels[idx]
        ax.scatter(gunler, [idx] * len(gunler), s=50, label=isim, c=cmap(seniority))

    ax.set_yticks(np.arange(len(doktor_isimleri)))
    ax.set_yticklabels(doktor_isimleri)
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xlabel("Günler")
    ax.set_ylabel("Doktorlar")
    ax.set_title("Doktorların Çalışma Takvimi")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()


plot_schedule()

import pandas as pd
import numpy as np

# Haftanın günleri
days_of_week_ordered = [
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
]

# her bir gün için haftanın hangi günü olduğunu belirleme
week_days_november = [(i + first_day_of_month) % 7 for i in range(n)]

# Sonuçları bir liste içinde toplama
results = []
for i in range(1, D + 1):
    for j in range(1, n + 1):
        if x[i, j].solution_value() > 0.5:
            day_of_week = days_of_week_ordered[week_days_november[j - 1]]
            results.append({"Doktor": f"Doktor {i}", "Gün": day_of_week})

# Sonuçların DataFrame olarak oluşturulması
df = pd.DataFrame(results)

# Pivot tablo oluşturma ve sıralama yapma
pivot_table = df.pivot_table(
    index="Doktor", columns="Gün", aggfunc="size", fill_value=0
)
pivot_table = pivot_table.reindex(columns=days_of_week_ordered)  # Günler için sıralama

# Hafta içi ve hafta sonu toplamlarını hesaplama
pivot_table["Weekdays"] = pivot_table[
    ["Pazartesi", "Salı", "Çarşamba", "Perşembe"]
].sum(axis=1)
pivot_table["Weekends"] = pivot_table[["Cuma", "Cumartesi", "Pazar"]].sum(axis=1)
pivot_table["Lecture Days"] = pivot_table["Salı"] + pivot_table["Perşembe"]


# Doktor isimlerini sayılara dönüştürme
pivot_table.index = pivot_table.index.str.replace("Doktor ", "").astype(int)

# Sayısal değerlere göre sıralama
pivot_table = pivot_table.sort_index()

# İndeks
pivot_table.index = "Doktor " + pivot_table.index.astype(str)

# Pivot tabloyu yazdırma
print(pivot_table)

# Her gün için kıdemlerde çalışan doktor sayısı
doktor_sayilari = np.zeros((n, len(kıdem_aralıkları)))

# Günlük toplam doktor sayısı
toplam_doktorlar = np.zeros(n)

# Takvim
takvim = {}
for gün in range(1, n + 1):
    takvim[gün] = f"{gün} {calendar.day_name[(calendar.weekday(2024, month, gün))]}"

for t in range(1, n + 1):
    for i in range(1, D + 1):
        for kıdem, (start, end) in enumerate(kıdem_aralıkları, 1):
            if start <= i <= end:
                doktor_sayilari[t - 1, kıdem - 1] += x[i, t].solution_value()
                toplam_doktorlar[t - 1] += x[i, t].solution_value()

# Görselleştirme
fig, ax = plt.subplots(figsize=(10, 10))

# Toplam doktorlar sütunu eklemek
doktor_sayilari = np.column_stack((doktor_sayilari, toplam_doktorlar))

sns.heatmap(
    doktor_sayilari,
    annot=True,
    cmap="YlGnBu",
    fmt="g",
    cbar_kws={"label": "Doctor Count"},
)

# Eksen etiketleri
etiketler = [f"Kıdem {kıdem}" for kıdem in range(1, 8)] + ["Toplam"]
ax.set_xticklabels(etiketler)
ax.set_yticklabels(
    [takvim[t] for t in range(1, n + 1)], rotation=0
)  # Rotate the date labels

# Renk çubuğu
cbar = ax.collections[0].colorbar
cbar.set_label("Doktor Sayısı")

plt.title("Kıdeme Göre Doktor Çalışma Dağılımı")
plt.xlabel("Kıdem")
plt.ylabel("Günler")

plt.show()
