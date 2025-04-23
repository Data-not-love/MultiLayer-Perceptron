import pandas as pd

#loại m sau các thuộc tính
#loạt tỷ sau giá
def process_data(input_file, output_file):
    # Đọc file CSV
    data = pd.read_csv(input_file)

    # Loại bỏ ký tự 'm' và chuyển đổi các giá trị thành kiểu số (float)
    data['Chiều ngang'] = data['Chiều ngang'].replace(r'm', '', regex=True).astype(float)
    data['Chiều dài'] = data['Chiều dài'].replace(r'm', '', regex=True).astype(float)
    data['Đường trước nhà'] = data['Đường trước nhà'].replace(r'm', '', regex=True).astype(float)
    data['Đường'] = data['Đường'].replace(r'Đường ', '', regex=True)
    data['Phường'] = data['Phường'].replace(r'Phường ', '', regex=True)
    data['Quận'] = data['Quận'].replace(r'Quận ', '', regex=True)




    # Lưu lại kết quả vào file mới
    data.to_csv(output_file, index=False)

process_data('F:/alonhadat.csv','F:/3.5 Years/First Year/Python/Mini_Linear/ultra_clean1.csv')

def xlsx_to_csv(input_file, output_file):
    # Đọc tệp Excel
    data = pd.read_excel(input_file, engine='openpyxl')

    # Ghi tệp CSV
    data.to_csv(output_file, index=False)

# File kết quả
#xlsx_to_csv('F:/3.5 Years/First Year/Python/Mini_Linear/alonhadat_output_single.csv','F:/3.5 Years/First Year/Python/Mini_Linear/ultra_clean1.csv')

