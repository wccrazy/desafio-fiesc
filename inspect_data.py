import pandas as pd
import json

CSV_PATH = "data/banner.csv"

def inspect():
    print("=" * 80)
    print("INSPEÇÃO DO DATASET:", CSV_PATH)
    print("=" * 80)

    # Leitura do dataset
    df = pd.read_csv(CSV_PATH)

    # 1. Total de linhas e colunas
    num_rows, num_cols = df.shape
    print(f"\n1. DIMENSÕES:")
    print(f"   Total de linhas: {num_rows:,}")
    print(f"   Total de colunas: {num_cols}")
    print(f"   Colunas: {list(df.columns)}")

    # 2. Intervalo de datas
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    min_date = df["created_at_dt"].min()
    max_date = df["created_at_dt"].max()
    print(f"\n2. INTERVALO DE DATAS (created_at):")
    print(f"   Data inicial (mínima): {min_date}")
    print(f"   Data final (máxima):   {max_date}")

    # 3. Contagem de registros por classe da coluna fault
    fault_counts = df["fault"].value_counts(dropna=False)
    print(f"\n3. CONTAGEM DE REGISTROS POR CLASSE (fault) - Total de classes: {len(fault_counts)}:")
    for fault, count in fault_counts.items():
        pct = (count / num_rows) * 100
        print(f"   - {str(fault):<35}: {count:>7,} ({pct:6.2f}%)")

    # 4. Valores únicos de rotação (rpm)
    unique_rpm = sorted(df["rpm"].dropna().unique())
    print(f"\n4. VALORES ÚNICOS DE ROTAÇÃO (rpm) - Total: {len(unique_rpm)}:")
    print(f"   {unique_rpm}")

    # 5. Busca pelo registro com id == 114387
    target_id = 114387
    record = df[df["id"] == target_id]
    print(f"\n5. BUSCA PELO REGISTRO id == {target_id}:")
    if not record.empty:
        print(f"   Registro encontrado com sucesso!")
        row_dict = record.iloc[0].drop(labels=["created_at_dt"], errors="ignore").to_dict()
        for k, v in row_dict.items():
            print(f"   - {k:<30}: {v}")
    else:
        print(f"   Registro com id {target_id} NÃO encontrado no dataset.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect()
