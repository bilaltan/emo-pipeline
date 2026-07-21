import pandas as pd
import openpyxl

excel_path = "results/ogbn-arxiv/ogbn-arxiv_results.xlsx"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    sheets = wb.sheetnames
    print(f"Arxiv Excel loaded successfully! Sheets found: {sheets}\n")
    
    if 'summary' in sheets:
        df = pd.read_excel(excel_path, sheet_name='summary')
        print("=== SUMMARY SHEET ===")
        print(df)
        print("\n" + "="*50 + "\n")
        
    for sheet in sheets:
        if sheet != 'summary':
            print(f"=== Sheet: {sheet} ===")
            df = pd.read_excel(excel_path, sheet_name=sheet)
            print(df.head(5))
            print("\n" + "-"*50 + "\n")
            
except Exception as e:
    print(f"Error loading workbook: {e}")
