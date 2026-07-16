import pandas as pd
import openpyxl

excel_path = "results/ogbn-products/ogbn-products_results.xlsx"

try:
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    sheets = wb.sheetnames
    print(f"Excel file loaded successfully! Sheets found: {sheets}\n")
    
    for sheet in sheets:
        print(f"=== Sheet: {sheet} ===")
        # Load sheet into pandas
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            print(df.head(20))
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"Error reading sheet {sheet}: {e}\n")
            
except Exception as e:
    print(f"Error loading workbook: {e}")
