import xarray as xr
import pandas as pd
import glob
import os
from config import PATHS

print("🔍 1. Mencari file data mentah...")
all_files = sorted(glob.glob(PATHS["hf_radar_pattern"]))
if os.path.exists(PATHS["nc_file_arus"]):
    all_files.append(PATHS["nc_file_arus"])

if not all_files:
    print("❌ File mentah tidak ditemukan!")
    exit()

print(f"🔄 2. Membuka {len(all_files)} file mentah...")
datasets_list = []
for f in all_files:
    try:
        ds_temp = xr.open_dataset(f, engine='netcdf4')[['u']].load()
        datasets_list.append(ds_temp)
    except:
        pass # Skip file rusak

# Gabungkan data mentah
ds = xr.concat(datasets_list, dim='time').sortby('time')
ds = ds.sel(time=~ds.get_index('time').duplicated())

print("📊 3. Menghitung ketersediaan titik spasial per jam...")
spatial_dims = [dim for dim in ds.dims if dim != 'time']
valid_points_per_time = ds['u'].notnull().sum(dim=spatial_dims).values
waktu = ds.time.values

# Masukkan ke DataFrame
df = pd.DataFrame({
    'waktu': pd.to_datetime(waktu),
    'jumlah_titik_valid': valid_points_per_time
})

# Anggap radar mati jika jumlah titik yang terekam sangat sedikit (misal < 10 titik)
# atau 0 sama sekali.
df_mati = df[df['jumlah_titik_valid'] == 0].copy()

if len(df_mati) == 0:
    print("\n✅ DATA MENTAH AMAN! Tidak ada jam di mana radar mati total (0 titik).")
    print("Artinya kerusakan murni karena parameter config.py yang terlalu ketat.")
else:
    df_mati['selisih'] = df_mati['waktu'].diff()
    df_mati['segmen_baru'] = df_mati['selisih'] > pd.Timedelta(hours=1)
    df_mati['id_segmen'] = df_mati['segmen_baru'].cumsum()
    
    print(f"\n⚠️ Total akumulasi radar mati dari data mentah: {len(df_mati)} jam")
    print("-" * 60)
    print("DAFTAR PERIODE RADAR MATI (Diurutkan dari durasi terlama):")
    
    rekap = df_mati.groupby('id_segmen').agg(
        awal=('waktu', 'min'),
        akhir=('waktu', 'max'),
        durasi_jam=('waktu', 'count')
    ).sort_values('durasi_jam', ascending=False)
    
    for idx, row in rekap.head(20).iterrows():
        print(f"Mati {row['durasi_jam']:>4} jam | {row['awal'].strftime('%Y-%m-%d %H:%M')} s/d {row['akhir'].strftime('%Y-%m-%d %H:%M')}")