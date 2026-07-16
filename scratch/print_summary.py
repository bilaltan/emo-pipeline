import pandas as pd

excel_path = "results/ogbn-products/ogbn-products_results.xlsx"

# Print all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_excel(excel_path, sheet_name='summary')
print(df)
