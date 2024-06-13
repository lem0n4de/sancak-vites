import numpy as np
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import pandas as pd
import random
import datetime


class Vites:
    __solver: pywraplp.Solver
    kıdem_aralıkları: list[tuple[int, int]]
    df: pd.DataFrame
    number_of_doctors: int
    # time related
    first_day_of_month: int
    number_of_days: int
    # variables
    dev: dict
    deveks: dict
    dev1: dict
    deveks1: dict
    nobet_listesi: dict

    def __init__(self) -> None:
        self.__solver = pywraplp.Solver.CreateSolver("SCIP")
        if not self.__solver:
            raise ValueError("SCIP çözücüsü bulunamadı!")

    def get_kıdem_aralıkları(self, kıdem_file: str = "asistan_kidem.xlsx"):
        self.df = pd.read_excel("asistan_kidem.xlsx")
        self.kıdem_aralıkları = []
        for i in set(self.df["Kıdem"]):
            aralık = min(self.df[self.df["Kıdem"] == i].index + 1), max(
                self.df[self.df["Kıdem"] == i].index + 1
            )
            self.kıdem_aralıkları.append(aralık)
        self.kıdem_aralıkları = self.kıdem_aralıkları[::-1]
        self.df["Ad-Soyad"] = self.df.apply(
            lambda row: row["Ad"] + " " + row["Soyad"], axis=1
        )

    def setup_month(
        self,
        year: int = datetime.date.today().year,
        month: int = datetime.date.today().month + 1,
    ):
        self.first_day_of_month = datetime.date(year, month, 1).weekday()
        self.number_of_days = calendar.monthrange(year, month)[1]
        self.number_of_doctors = len(self.df)

    def setup_variables(self):
        # Lecture day's deviations
        self.dev = {}
        for i in range(1, self.number_of_doctors + 1):
            for j in range(1, self.number_of_days + 1):
                self.dev[i, j] = self.__solver.IntVar(0, 1, f"dev[{i}][{j}]")
        self.deveks = {}
        for i in range(1, self.number_of_doctors + 1):
            for j in range(1, self.number_of_days + 1):
                self.deveks[i, j] = self.__solver.IntVar(0, 1, f"deveks[{i}][{j}]")

        # Sapma değerleri için değişkenlerin tanımlanması ilk 3 kıdem için için
        self.dev1 = {}
        for i in range(1, self.number_of_doctors + 1):
            for j in range(1, self.number_of_days + 1):
                self.dev1[i, j] = self.__solver.IntVar(0, 1, f"dev1[{i}][{j}]")
        self.deveks1 = {}
        for i in range(1, self.number_of_doctors + 1):
            for j in range(1, self.number_of_days + 1):
                self.deveks1[i, j] = self.__solver.IntVar(0, 1, f"deveks1[{i}][{j}]")

        # Değişkenlerin tanımlanması
        self.nobet_listesi = {}
        for i in range(1, self.number_of_doctors + 1):
            for j in range(1, self.number_of_days + 1):
                self.nobet_listesi[i, j] = self.__solver.BoolVar(
                    f"nobet_listesi[{i}][{j}]"
                )

    def setup_kidem_rules(self):
        # Her gün her kıdemden en az bir doktor
        for j in range(1, self.number_of_days + 1):
            for aralık in self.kıdem_aralıkları:
                start, end = aralık
                if (end - start) <= 2:
                    self.__solver.Add(
                        sum(self.nobet_listesi[i, j] for i in range(start, end + 1))
                        <= 1
                    )  # eğer kıdemde yeterli kişi yoksa o kıdemden nöbet tutan kimse olmayabilir(örneğin kıdemdeki kişi sayısı 3 kişi ve altındaysa)
                else:
                    self.__solver.Add(
                        1
                        <= sum(self.nobet_listesi[i, j] for i in range(start, end + 1))
                    )  # o gün için o kıdemden çalışacak bütün doktorların toplamının sayısının 1 den büyük olması

        # 6. kıdemi 4ten küçük yapma
        for j in range(1, self.number_of_days + 1):
            start, end = (23, 33)
            self.__solver.Add(
                sum(self.nobet_listesi[i, j] for i in range(start, end + 1)) <= 3
            )
            self.__solver.Add(
                sum(self.nobet_listesi[i, j] for i in range(start, end + 1)) >= 2
            )
        """
        #7. kıdem
        for j in range(1, self.number_of_days+1):
        start, end = (34,36)
        self.__solver.Add(sum(self.nobet_listesi[i, j] for i in range(start, end+1)) <= 2)
        self.__solver.Add(sum(self.nobet_listesi[i, j] for i in range(start, end+1)) >= 1)
        """
        # 4. kıdemi 3ten küçük yapma
        for j in range(1, self.number_of_days + 1):
            start, end = (13, 17)
            self.__solver.Add(
                sum(self.nobet_listesi[i, j] for i in range(start, end + 1)) <= 2
            )
        """
        #5. kıdemi 3ten küçük yapma
        for j in range(1, self.number_of_days+1):
        start, end = (18, 22)
        self.__solver.Add(sum(self.nobet_listesi[i, j] for i in range(start, end+1)) <= 3)
        """

        # iki günde bir nöbet -bir nöbetten  sonra iki gün boş kalmalı-
        for i in range(1, self.number_of_doctors + 1):
            for j in range(1, self.number_of_days - 1):
                self.__solver.Add(
                    self.nobet_listesi[i, j]
                    + self.nobet_listesi[i, j + 1]
                    + self.nobet_listesi[i, j + 2]
                    <= 1
                )

    def setup_month_start_end_rules(self):
        # bir önceki ayın son günü (last_day) ve sondan bir önceki günü (second_last_day) nöbetten sonra 2 gün dinlenme kısıtı
        worked_last_day = {1, 6, 7, 8, 9, 15, 19, 28, 32}
        worked_second_last_day = {2, 5, 12, 16, 17, 18, 27, 31}

        # Constraints based on the last two days of the previous month
        for i in range(1, self.number_of_doctors + 1):
            if i in worked_last_day:
                # Doctor cannot work on the 1st and 2nd days of the current month
                self.__solver.Add(self.nobet_listesi[i, 1] == 0)
                self.__solver.Add(self.nobet_listesi[i, 2] == 0)
            elif i in worked_second_last_day:
                # Doctor cannot work on the 1st day of the current month
                self.__solver.Add(self.nobet_listesi[i, 1] == 0)

    def setup_average_doctor_per_day(self):
        # Bir gündeki doktor sayısının olabildiğince eşit olması  (ortalamadan -1 olması kısıtı) bir günde 9 veya 10 doktor bulunması gibi
        for j in range(1, self.number_of_days + 1):
            self.__solver.Add(
                sum(
                    self.nobet_listesi[i, j]
                    for i in range(1, self.number_of_doctors + 1)
                )
                <= round(self.number_of_doctors * 8 / self.number_of_days)
            )
            self.__solver.Add(
                round(self.number_of_doctors * 8 / self.number_of_days) - 1
                <= sum(
                    self.nobet_listesi[i, j]
                    for i in range(1, self.number_of_doctors + 1)
                )
            )

        # Bir gündeki doktor sayısının olabildiğince eşit olması  (ortalamadan  +1 olması kısıtı)

        # for j in range(1, self.number_of_days+1):
        #     solver.Add(sum(self.nobet_listesi[i, j] for i in range(1, self.number_of_doctors+1)) <= round(self.number_of_doctors*8/self.number_of_days)+1)
        #     solver.Add(round(self.number_of_doctors*8/self.number_of_days) <= sum(self.nobet_listesi[i, j] for i in range(1, self.number_of_doctors+1)))

    def setup_equal_weekend_shifts(self):
        # Hafta içi ve hafta sonu nöbetlerinin eşit ya da olabildiğince eşit dağılması
        weekdays = [0, 1, 2, 3]  # Pazartesi, Salı, Çarşamba, Perşembe
        weekends = [4, 5, 6]  # Cuma, Cumartesi, Pazar

        for i in range(1, self.number_of_doctors + 1):
            weekday_shifts = sum(
                self.nobet_listesi[i, j]
                for j in range(1, self.number_of_days + 1)
                if (j + self.first_day_of_month - 1) % 7 in weekdays
            )
            weekend_shifts = sum(
                self.nobet_listesi[i, j]
                for j in range(1, self.number_of_days + 1)
                if (j + self.first_day_of_month - 1) % 7 in weekends
            )
            self.__solver.Add(weekday_shifts - weekend_shifts <= 2)
            self.__solver.Add(-2 <= weekday_shifts - weekend_shifts)

    def setup_equal_lesson_days(self):
        self.salı_günleri = [
            j
            for j in range(1, self.number_of_days + 1)
            if (j + self.first_day_of_month - 1) % 7 == 1
        ]  # Salı
        self.perşembe_günleri = [
            j
            for j in range(1, self.number_of_days + 1)
            if (j + self.first_day_of_month - 1) % 7 == 3
        ]  # Perşembe

        # Her doktorun toplam Salı ve Perşembe günleri tuttuğu nöbet sayılarının olabildiğince eşit olması kısıtı
        for i in range(1, self.number_of_doctors + 1):
            salı_perşembe_toplam_i = sum(
                self.nobet_listesi[i, gün] for gün in self.salı_günleri
            ) + sum(self.nobet_listesi[i, gün] for gün in self.perşembe_günleri)
            deviation = sum(
                self.dev[i, gün] for gün in (self.salı_günleri + self.perşembe_günleri)
            ) - sum(
                self.deveks[i, gün]
                for gün in (self.salı_günleri + self.perşembe_günleri)
            )
            self.__solver.Add(salı_perşembe_toplam_i + deviation == 2)

    def setup_shift_count(self):
        # her doktor 8 nöbet tutmalı
        for i in range(1, self.number_of_doctors + 1):
            # if i == 24:
            #   solver.Add(sum(self.nobet_listesi[i, j] for j in range(1, self.number_of_days+1)) == 2) #özel durum (24. doktor 2 nöbet tutacak)
            if i == 23:
                self.__solver.Add(
                    sum(
                        self.nobet_listesi[i, j]
                        for j in range(1, self.number_of_days + 1)
                    )
                    == 0
                )
            elif i == 5:
                self.__solver.Add(
                    sum(
                        self.nobet_listesi[i, j]
                        for j in range(1, self.number_of_days + 1)
                    )
                    == 7
                )
            else:
                self.__solver.Add(
                    sum(
                        self.nobet_listesi[i, j]
                        for j in range(1, self.number_of_days + 1)
                    )
                    == 8
                )

    def setup_shift_on_day(self, avail: dict[int, list[int]]):
        # each doctor works on their available dates
        for doctor, available_dates in avail.items():
            for date in available_dates:
                self.__solver.Add(
                    self.nobet_listesi[doctor, date] == 1
                )  # Doctor must work on their available dates

    def setup_shift_off_day(self, non_avail: dict[int, list[int]]):
        # each doctor doesn't work on their unavailable dates
        for doctor, unavailable_dates in non_avail.items():
            for date in unavailable_dates:
                self.__solver.Add(
                    self.nobet_listesi[doctor, date] == 0
                )  # Doctor cannot work on their unavailable dates

    def setup_couple_shifts(self, evli_çiftler):
        # Her çift için, ay boyunca 6 gün birlikte ve 2 gün ayrı nöbet tutma kısıtları
        for doktor1, doktor2 in evli_çiftler:
            # Her gün için birlikte nöbet tutup tutmadıklarını belirten değişkenler
            birlikte_nöbet = [
                self.__solver.BoolVar(f"birlikte_nöbet[{doktor1}][{doktor2}][{j}]")
                for j in range(1, self.number_of_days + 1)
            ]

            for j in range(1, self.number_of_days + 1):
                # Eğer her iki doktor da aynı gün nöbetçi ise, birlikte_nöbet[j] 1 olur
                self.__solver.Add(
                    birlikte_nöbet[j - 1] <= self.nobet_listesi[doktor1, j]
                )
                self.__solver.Add(
                    birlikte_nöbet[j - 1] <= self.nobet_listesi[doktor2, j]
                )
                self.__solver.Add(
                    birlikte_nöbet[j - 1]
                    >= self.nobet_listesi[doktor1, j]
                    + self.nobet_listesi[doktor2, j]
                    - 1
                )

            # Her çift için toplam 6 gün birlikte nöbet tutma kısıtı
            self.__solver.Add(sum(birlikte_nöbet) == 6)

    def setup(self, avail, non_avail, evli_ciftler):
        self.get_kıdem_aralıkları()
        self.setup_month()
        self.setup_variables()
        self.setup_kidem_rules()
        self.setup_month_start_end_rules()
        self.setup_average_doctor_per_day()
        self.setup_equal_weekend_shifts()
        self.setup_equal_lesson_days()
        self.setup_shift_count()
        self.setup_shift_on_day(avail)
        self.setup_shift_off_day(non_avail)
        self.setup_couple_shifts(evli_ciftler)

        # Hedef fonksiyonu tanımlama
        objective_terms = []
        for doktor in range(1, self.number_of_doctors + 1):
            for gün in self.salı_günleri + self.perşembe_günleri:
                objective_terms.append(self.dev[doktor, gün])
                objective_terms.append(self.deveks[doktor, gün])

        self.__solver.Minimize(self.__solver.Sum(objective_terms))

    def create_shifts(self):
        status = self.__solver.Solve()
        # Çözümün yazdırılması
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:

            for i in range(1, self.number_of_doctors + 1):
                for j in range(1, self.number_of_days + 1):
                    if self.nobet_listesi[i, j].solution_value() > 0.5:
                        print(
                            self.df["Ad-Soyad"].iloc[i - 1]
                            + f", Gün {j} için nöbettedir."
                        )
        else:
            print("Optimal çözüm bulunamadı!")


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

evli_çiftler = [(1, 6), (2, 11), (16, 17)]  # Çiftlerin doktor numaraları
vites = Vites()
vites.setup(doctor_availabilities, doctor_non_availabilities, evli_çiftler)
vites.create_shifts()


import sys

sys.exit(0)

from collections import defaultdict
import pandas as pd

# Boş bir DataFrame
df1 = pd.DataFrame()

# Ayın günleri
days = [f"{day:02d}.4.2024" for day in range(1, 32)]  # aydan aya değişmeli
df1["Days"] = days


doktor_nobetleri = defaultdict(list)

for i in range(1, self.number_of_doctors + 1):
    # calisma = []
    # for j in range(1, self.number_of_days + 1):
    #     if self.nobet_listesi[i, j].solution_value() > 0.5:
    #         calisma.append(j)
    # doktor_calisma_gunleri.append(calisma)

    for j in range(1, self.number_of_days + 1):
        if self.nobet_listesi[i, j].solution_value() > 0.5:
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
    for i in range(1, self.number_of_doctors + 1):
        calisma = []
        for j in range(1, self.number_of_days + 1):
            if self.nobet_listesi[i, j].solution_value() > 0.5:
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
    ax.set_xticks(np.arange(1, self.number_of_days + 1))
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
week_days_november = [(i + first_day_of_month) % 7 for i in range(self.number_of_days)]

# Sonuçları bir liste içinde toplama
results = []
for i in range(1, self.number_of_doctors + 1):
    for j in range(1, self.number_of_days + 1):
        if self.nobet_listesi[i, j].solution_value() > 0.5:
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
doktor_sayilari = np.zeros((self.number_of_days, len(kıdem_aralıkları)))

# Günlük toplam doktor sayısı
toplam_doktorlar = np.zeros(self.number_of_days)

# Takvim
takvim = {}
for gün in range(1, self.number_of_days + 1):
    takvim[gün] = f"{gün} {calendar.day_name[(calendar.weekday(2024, month, gün))]}"

for t in range(1, self.number_of_days + 1):
    for i in range(1, self.number_of_doctors + 1):
        for kıdem, (start, end) in enumerate(kıdem_aralıkları, 1):
            if start <= i <= end:
                doktor_sayilari[t - 1, kıdem - 1] += self.nobet_listesi[
                    i, t
                ].solution_value()
                toplam_doktorlar[t - 1] += self.nobet_listesi[i, t].solution_value()

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
    [takvim[t] for t in range(1, self.number_of_days + 1)], rotation=0
)  # Rotate the date labels

# Renk çubuğu
cbar = ax.collections[0].colorbar
cbar.set_label("Doktor Sayısı")

plt.title("Kıdeme Göre Doktor Çalışma Dağılımı")
plt.xlabel("Kıdem")
plt.ylabel("Günler")

plt.show()
