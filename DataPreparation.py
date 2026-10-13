import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.parquet as pq

# %% Przydatne ustawienia
# Wyświetl wszystkie kolumny
pd.set_option("display.max_columns", None)

# Wyświetl wszystkie wiersze
#pd.set_option("display.max_rows", None)

# Nie skracaj zawartości komórek
pd.set_option("display.max_colwidth", None)

# Nie ograniczaj szerokości tabeli
pd.set_option("display.width", None)


# %% Wczytanie danych
#to nie działa bo xml jest za duzy
#data_df = pd.read_xml("data/players_list_xml_foa.xml")
#print(data_df.head())


# %% Wczytanie czesci danych xml
#To działa dla 1000 wierszy
players = []
context = ET.iterparse("data/players_list_xml_foa.xml", events=("start", "end"))
_, root = next(context)
for event, elem in context:
    if event == "end" and elem.tag == "player":
        players.append({
            child.tag: child.text
            for child in elem
        })
        root.clear()
        if len(players) == 1000:
            break
data_df = pd.DataFrame(players)
print(data_df.head())
print(data_df.shape)
print(data_df.head().T)


# %% Przetworzenie do pliku typu Parquet

# Ścieżki
xml_file = "data/players_list_xml_foa.xml"
parquet_file = "data/players.parquet"

# Konwersja XML do Parquet
batch_size = 10000
batch = []
writer = None
columns = None
schema = None
count = 0

context = ET.iterparse(xml_file, events=("start", "end"))
_, root = next(context)

try:
    for event, elem in context:
        if event == "end" and elem.tag == "player":
            player = {
                child.tag: child.text
                for child in elem
            }
            if writer is None:
                columns = list(player.keys())
                schema = pa.schema([
                    (col, pa.string()) for col in columns
                ])
                writer = pq.ParquetWriter(
                    parquet_file,
                    schema=schema,
                    compression="zstd"
                )
            batch.append(player)
            count += 1
            elem.clear()
            root.clear()
            if len(batch) >= batch_size:
                table = pa.Table.from_pydict(
                    {
                        col: [row.get(col) for row in batch]
                        for col in columns
                    },
                    schema=schema
                )
                writer.write_table(table)
                batch.clear()
                print(f"Przetworzono: {count} zawodników")
    # Zapis ostatniej, niepełnej partii
    if batch and writer is not None:
        table = pa.Table.from_pydict(
            {
                col: [row.get(col) for row in batch]
                for col in columns
            },
            schema=schema
        )
        writer.write_table(table)
finally:
    if writer is not None:
        writer.close()
print(f"Zakończono. Łącznie zawodników: {count}")


# %% Wczytanie etykiet

with open("data/Legend.txt", encoding="utf-8") as f:
    labels = dict(
        line.strip().split(" - ", 1)
        for line in f
        if " - " in line
    )
print(labels)


# %% Wczytanie danych z Parquet

data_df = pd.read_parquet("data/players.parquet")
print(data_df.head())
print(data_df.shape)


# %% dodanie etykiet do danych

data_view = data_df.copy()
data_view.columns = pd.MultiIndex.from_tuples(
    [(col, labels.get(col, "")) for col in data_df.columns]
)
print(data_view.head())

# %% Kontrola etykiet
print(list(labels.items())[:10])

print("Kolumny bez etykiet:")
print([col for col in data_df.columns if col not in labels])

# Naprawa etykiet
# %% Mapowanie nazw XML na etykiety z TXT

xml_to_legend = {
    "rating": "STD/SRTNG",
    "rapid_rating": "RPD/RRTNG",
    "blitz_rating": "BLZ/BRTNG",
    "games": "SGM",
    "rapid_games": "RGM",
    "blitz_games": "BGM",
    "k": "SK",
    "rapid_k": "RK",
    "blitz_k": "BK",
    "birthday": "B-day/BORN"
}
# %% Pozostałe etykiety
extra_labels = {
    "fideid": "FIDE player ID",
    "name": "Player name",
    "country": "Federation",
    "sex": "Sex",
    "title": "FIDE title",
    "w_title": "Women's FIDE title",
    "o_title": "Other title",
    "foa_title": "FIDE Online Arena title",
    "flag": "Player status flag"
}
# %% Dodanie opisów pod nazwami kolumn
column_labels = {
    **extra_labels,
    **{
        col: labels[legend_name]
        for col, legend_name in xml_to_legend.items()
        if legend_name in labels
    }
}
data_view = data_df.copy(deep=False)
data_view.columns = pd.MultiIndex.from_tuples(
    [
        (col, column_labels.get(col, "Brak opisu"))
        for col in data_df.columns
    ]
)
print(data_view.head())
# %% Kontrola opisów
print("Liczba zmiennych:", len(data_df.columns))
print("Liczba dopasowanych etykiet:", len(column_labels))

print("Zmienne bez opisu:")
print([col for col in data_df.columns if col not in column_labels])


#############################
# Sprawdzenie jakości danych do czyszczenia
# %%
print(data_df.dtypes)
print(data_df.isna().sum())
print(data_df.isnull().sum())
print(data_df.shape)
print(data_df.describe())


## Konwersja danych
# %% Konwersja typów zmiennych

numeric_cols = [
    "rating", "games", "k",
    "rapid_rating", "rapid_games", "rapid_k",
    "blitz_rating", "blitz_games", "blitz_k",
    "birthday"
]

data_df[numeric_cols] = data_df[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
).astype("Int32")
print(data_df.dtypes)
