"""ETL pipeline untuk dataset automobileEDA_dirty_training.csv.

Extract   : baca CSV mentah dari data/raw/
Transform : inspeksi -> cleaning -> transformation
Load      : simpan hasil ke data/processed/automobileEDA_processed.csv

Jalankan dari root project:  python src/pipeline.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "automobileEDA_processed.csv"

# --- Konfigurasi transformasi -----------------------------------------------
# Min-Max Scaling hanya untuk kolom numerik KONTINU. Sengaja TIDAK di-scale:
#   - symboling     : rating risiko ordinal (-2..3), bukan ukuran kontinu
#   - length, width : sudah berskala 0-1 dari sumber dataset
#   - diesel, gas   : kolom biner 0/1
# Catatan: price adalah kolom target. Di tugas ini price tetap di-scale
# (price_scaled) sebagai bahan latihan, dan kolom price asli tetap disimpan.
# Untuk modeling, target umumnya tidak di-scale.
SCALE_COLS = [
    "normalized-losses", "wheel-base", "height", "curb-weight", "engine-size",
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "city-L/100km", "price",
]
ORDINAL_COL = "horsepower-binned"            # kategori berurutan
ORDINAL_ORDER = ["Low", "Medium", "High"]
ONEHOT_COLS = ["aspiration", "body-style", "drive-wheels"]   # kategori tanpa urutan

CATEGORICAL_COLS = [
    "make", "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "engine-type", "num-of-cylinders", "fuel-system",
    "horsepower-binned",
]


def line(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ----------------------------------------------------------------- EXTRACT
def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Membaca dataset mentah. File raw tidak pernah ditulis ulang."""
    df = pd.read_csv(path)
    print(f"[EXTRACT] Dataset dibaca dari: {path}")
    return df


# --------------------------------------------------------------- INSPECTION
def inspect_data(df: pd.DataFrame) -> None:
    """Menampilkan kondisi awal dataset ke terminal."""
    line("DATA INSPECTION")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("Lima baris pertama:\n", df.head())
    print(f"\nJumlah baris x kolom : {df.shape[0]} x {df.shape[1]}")
    print("\nNama kolom:", list(df.columns))
    print("\nTipe data:\n", df.dtypes.to_string())
    miss = df.isna().sum()
    print("\nMissing values (hanya kolom yang kosong):\n", miss[miss > 0].to_string())
    print(f"\nTotal missing values : {int(miss.sum())}")
    print(f"Duplicate records    : {int(df.duplicated().sum())}")
    print("\nNilai unik kolom kategorikal:")
    for col in CATEGORICAL_COLS:
        print(f"  {col}: {sorted(df[col].dropna().unique().tolist())}")
    print("\nContoh format transaction_date:", df["transaction_date"].dropna().sample(8, random_state=1).tolist())


# ----------------------------------------------------------------- CLEANING
def _parse_date(value):
    """Menyeragamkan 4 format tanggal yang ada di dataset.

    - 2025-01-01   : ISO (yyyy-mm-dd)
    - 02/01/2025   : dd/mm/yyyy  (tanda '/')
    - 01-03-2025   : mm-dd-yyyy  (tanda '-')
    - 04-Jan-2025  : dd-Mon-yyyy
    Aturan slash vs dash ditentukan dari urutan tanggal harian pada data
    (baris ke-n = 1 Jan 2025 + n hari), yang terkonfirmasi untuk semua baris unik.
    """
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return pd.to_datetime(s, format="%Y-%m-%d")
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        return pd.to_datetime(s, format="%d/%m/%Y")
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s):
        return pd.to_datetime(s, format="%m-%d-%Y")
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data berdasarkan temuan pada tahap inspeksi."""
    line("DATA CLEANING")
    before = df.copy()
    df = df.copy()
    n_before = len(df)
    miss_before = df.isna().sum()

    # 1. Hapus spasi berlebih & samakan huruf pada kolom teks kategorikal
    for col in CATEGORICAL_COLS:
        s = df[col].astype("string").str.strip().str.lower()
        s = s.replace({"peugot": "peugeot"})  # salah ketik pada merek
        if col == "horsepower-binned":
            s = s.str.capitalize()  # Low / Medium / High
        df[col] = s.astype(object).where(s.notna(), np.nan)

    # 2. Perbaiki tipe data tanggal (string campuran -> datetime)
    df["transaction_date"] = df["transaction_date"].map(_parse_date)

    # 3. Hapus duplicate records (setelah penyeragaman agar duplikat tersembunyi ikut terdeteksi)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()  # index asli dipertahankan untuk pelacakan perubahan

    # 4. Missing values
    #    a. make kosong pada baris yang sama dengan duplikat -> tidak bisa diisi
    #       dengan akurat; tandai 'unknown' agar baris tetap dipertahankan.
    df["make"] = df["make"].fillna("unknown")
    #    b. num-of-doors (kategorikal) -> modus per body-style
    mode_by_body = df.groupby("body-style")["num-of-doors"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    df["num-of-doors"] = df["num-of-doors"].fillna(df["body-style"].map(mode_by_body))
    #    c. numerik (price, stroke, horsepower) -> median (tahan terhadap outlier)
    medians = {}
    for col in ["price", "stroke", "horsepower"]:
        medians[col] = df[col].median()
        df[col] = df[col].fillna(medians[col])
    #    d. horsepower-binned -> hitung ulang dari horsepower dengan batas yang
    #       dipakai dataset (Low <=101, Medium 102-155, High >=156)
    bins = [-np.inf, 101, 155, np.inf]
    df["horsepower-binned"] = pd.cut(
        df["horsepower"], bins=bins, labels=["Low", "Medium", "High"]
    ).astype(object)
    #    e. transaction_date kosong dipertahankan sebagai NaT (tidak bisa ditebak,
    #       dan tidak dipakai dalam perhitungan lain)

    # Kolom yang berubah: bandingkan nilai akhir dengan nilai mentah pada baris yang sama
    raw_aligned = before.loc[df.index]
    changed_cols = []
    for c in df.columns:
        old, new = raw_aligned[c], df[c]
        same = (old.astype(str) == new.astype(str)) | (old.isna() & new.isna())
        if c == "transaction_date":
            same = pd.Series(False, index=df.index)  # tipe berubah: string -> datetime
        if not same.all():
            changed_cols.append(c)
    df = df.reset_index(drop=True)

    miss_after = df.isna().sum()
    print(f"Jumlah baris sebelum cleaning : {n_before}")
    print(f"Jumlah baris sesudah cleaning : {len(df)}")
    print(f"Duplicate records dihapus     : {n_dup}")
    print("Nilai median pengisi          :", {k: float(v) for k, v in medians.items()})
    print(f"Total missing sebelum / sesudah: {int(miss_before.sum())} / {int(miss_after.sum())}")
    cmp = pd.DataFrame({"sebelum": miss_before, "sesudah": miss_after})
    print("\nMissing values per kolom (yang terdampak):\n", cmp[(cmp.sebelum > 0) | (cmp.sesudah > 0)].to_string())
    print("\nKolom yang mengalami perubahan:", changed_cols)
    return df


# ---------------------------------------------------------- TRANSFORMATION
def build_preprocessor() -> Pipeline:
    """Membangun preprocessing pipeline sklearn (Pipeline + ColumnTransformer).

    - MinMaxScaler(0, 1)  : fitur numerik kontinu (SCALE_COLS)
    - OrdinalEncoder      : horsepower-binned (Low < Medium < High)
    - OneHotEncoder       : aspiration, body-style, drive-wheels

    clip=True menjaga data baru yang nilainya di luar rentang data latih tetap
    berada di 0-1. handle_unknown="ignore" mencegah error bila ada kategori baru.
    Objek ini bisa dipakai ulang: fit pada data latih, transform pada data baru.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("scale", MinMaxScaler(feature_range=(0, 1), clip=True), SCALE_COLS),
            ("ordinal", OrdinalEncoder(categories=[ORDINAL_ORDER], dtype=int), [ORDINAL_COL]),
            ("onehot", OneHotEncoder(sparse_output=False, dtype=int, handle_unknown="ignore"), ONEHOT_COLS),
        ],
        remainder="drop",  # kolom asli tetap ada di dataframe asal (untuk perbandingan)
    )
    pipe = Pipeline(steps=[("preprocess", column_transformer)])
    return pipe.set_output(transform="pandas")


def _output_name(name: str) -> str:
    """Nama kolom keluaran ColumnTransformer -> nama kolom di dataset akhir."""
    step, col = name.split("__", 1)
    if step == "scale":
        return f"{col}_scaled"
    if step == "ordinal":
        return f"{col}_encoded"
    return col  # one-hot sudah berbentuk <kolom>_<nilai>


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transformasi data bersih: Min-Max scaling, ordinal encoding, one-hot encoding."""
    line("DATA TRANSFORMATION")
    df = df.copy()

    # Validasi: scaler butuh data tanpa NaN (sudah dijamin oleh tahap cleaning)
    n_nan = int(df[SCALE_COLS].isna().sum().sum())
    if n_nan:
        raise ValueError(f"Masih ada {n_nan} NaN pada kolom yang akan di-scale")

    preprocessor = build_preprocessor()
    features = SCALE_COLS + [ORDINAL_COL] + ONEHOT_COLS
    transformed = preprocessor.fit_transform(df[features])
    transformed.columns = [_output_name(c) for c in transformed.columns]

    scaled_cols = [f"{c}_scaled" for c in SCALE_COLS]
    if not transformed[scaled_cols].apply(lambda x: x.between(0, 1).all()).all():
        raise ValueError("Ada nilai hasil scaling di luar rentang 0-1")

    # Statistik sebelum vs sesudah Min-Max Scaling
    stats = pd.DataFrame({
        "min_sebelum": df[SCALE_COLS].min(),
        "max_sebelum": df[SCALE_COLS].max(),
        "min_sesudah": transformed[scaled_cols].min().set_axis(SCALE_COLS),
        "max_sesudah": transformed[scaled_cols].max().set_axis(SCALE_COLS),
    })
    print("Kolom di-scale (MinMaxScaler):", SCALE_COLS)
    print("Tidak di-scale: symboling, length, width, diesel, gas")
    print("\nStatistik SEBELUM vs SESUDAH Min-Max Scaling:")
    print(stats.round(3).to_string())

    result = pd.concat([df, transformed], axis=1)

    print("\nKolom baru:", list(transformed.columns))
    print("\nPerbandingan sebelum vs sesudah (5 baris pertama):")
    show = result[["price", "price_scaled", "horsepower", "horsepower_scaled", "curb-weight", "curb-weight_scaled",
                   "engine-size", "engine-size_scaled", "horsepower-binned",
                   "horsepower-binned_encoded", "body-style"]].head(5)
    print(show.to_string())
    return result


# -------------------------------------------------------------------- LOAD
def save_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Menyimpan dataset hasil olahan sebagai file baru."""
    line("LOAD")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset hasil olahan disimpan di: {path}  ({df.shape[0]} baris x {df.shape[1]} kolom)")


# ---------------------------------------------------------------- PIPELINE
def run_pipeline() -> pd.DataFrame:
    raw = load_data()
    inspect_data(raw)
    cleaned = clean_data(raw)
    transformed = transform_data(cleaned)
    save_data(transformed)
    return transformed


if __name__ == "__main__":
    run_pipeline()
