#!/usr/bin/env python3

'''
python find_cve_record.py --csv cve_without_descriptions_main_filled_without_model.csv 
'''
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Find and print a CVE record from CSV")
    parser.add_argument("--csv", required=True, help="Path to the CSV file (e.g. cve_without_descriptions_main.csv)")
    args = parser.parse_args()

    # Load CSV (only once)
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, dtype=str).fillna("")

    print(f"Loaded {len(df)} rows. You can now search by CVE ID (e.g., CVE-2017-17215).")
    print("Type 'exit' to quit.\n")

    while True:
        cve_id = input("Enter CVE ID: ").strip()
        if cve_id.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not cve_id:
            continue

        # Case-insensitive search for CVE
        result = df[df["cve"].str.lower() == cve_id.lower()]

        if result.empty:
            print(f"[!] No record found for {cve_id}\n")
        else:
            print(f"\n--- Found {len(result)} record(s) for {cve_id} ---")
            for _, row in result.iterrows():
                print("-" * 60)
                for col in df.columns:
                    val = str(row[col]).strip()
                    if val:
                        print(f"{col}: {val}")
                print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
