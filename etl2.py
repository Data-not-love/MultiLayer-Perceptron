import pandas as pd


# csv to xlsx
def csv_to_xls(input_file, output_file):
    data = pd.read_csv(input_file)

    data.to_excel(output_file, index=False)



# Gọi hàm với tên file đầu vào và đầu ra
input_file = 'F:/3.5 Years/First Year/Python\Mini_Linear/alonhadat.csv'  # Thay bằng đường dẫn file CSV của bạn
output_file = 'F:/3.5 Years/First Year/Python/Mini_Linear/test.xlsx'  # Kết quả lưu vào file Excel
csv_to_xls(input_file, output_file)
