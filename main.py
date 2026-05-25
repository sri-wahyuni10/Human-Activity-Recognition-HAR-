import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('time_series_data_human_activities.csv')

#tampilkan ukuran data
print("Ukuran data:", data.shape)

data_duplicate = data.duplicated().sum()
#print("Jumlah data duplikat:", data_duplicate)
data_clean = data.drop_duplicates(subset =['user', 'timestamp'],keep='first')
#print("Ukuran data setelah menghapus duplikat:", data_clean.shape)

#cek data kosong jika ada data kosong maka gunakan fungsi interpolate untuk mengisi data kosong
#data_clean = data_clean.interpolate(method='linear', limit_direction='forward', axis=0)

#Algoritma Magnitudo (Akar Kuadrat Jumlah Kuadrat)
data_clean['magnitude'] = np.sqrt(data_clean['x-axis']**2 + data_clean['y-axis']**2 + data_clean['z-axis']**2)

#data_clean.to_csv('data_clean.csv', index=False)

#Segmentasi Makro Aktivitas (Binning/Categorization)
#Kelompokkan aktivitas manusia menjadi dua kategori besar: Statis (diam/duduk/tidur) dan Dinamis (bergerak/berjalan/naik tangga) untuk melihat karakteristik umum sensor.
kategori_mapping = {
    'Sitting': 'Statis',
    'Standing': 'Statis',
    'Lying': 'Statis',
    'Walking': 'Dinamis',
    'Running': 'Dinamis',
    'Stairs': 'Dinamis'
}
data_clean['activity_category'] = data_clean['activity'].map(kategori_mapping)

#print(data_clean[['activity', 'activity_category']])
#Visualisasi Data

#Melihat pola aktivitas manusia berdasarkan magnitudo sensor dari waktu ke waktu menggunakan grafik garis untuk melihat bagaimana magnitudo berubah selama periode pengamatan.
plt.figure(figsize=(12, 6))
for category in data_clean['activity_category'].unique():
    subset = data_clean[data_clean['activity_category'] == category]
    plt.plot(subset['timestamp'], subset['magnitude'], label=category)
plt.title('Pola Aktivitas Manusia Berdasarkan Magnitudo Sensor dari Waktu ke Waktu')
plt.xlabel('Timestamp')
plt.ylabel('Magnitudo Sensor')
plt.legend()
plt.show()

#melihat hubungan magnitude dengan activity manusia menggunakan boxplot untuk melihat distribusi magnitudo sensor untuk setiap kategori aktivitas.
plt.figure(figsize=(8, 6))
data_clean.boxplot(column='magnitude', by='activity_category')
plt.title('Distribusi Magnitudo Sensor untuk Setiap Kategori Aktivitas')
plt.suptitle('')
plt.xlabel('Kategori Aktivitas')
plt.ylabel('Magnitudo Sensor')
plt.show()


#Analisis Korelasi
#Analisis korelasi antara magnitudo sensor dan kategori aktivitas untuk melihat apakah ada hubungan yang signifikan antara keduanya.
correlation = data_clean['magnitude'].corr(data_clean['activity_category'].astype('category').cat.codes)
print("Korelasi antara Magnitudo Sensor dan Kategori Aktivitas:", correlation)


    

