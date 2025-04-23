import pandas as pd
import numpy as np

def convert_real_estate_data(input_file, output_file):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(input_file)

    # Loại bỏ cột 'URL' nếu tồn tại
    if 'URL' in df.columns:
        df.drop(columns=['URL'], inplace=True)

    # Chuyển đổi cột 'Giá' từ '6,9 tỷ' thành 6.9
    df['Giá'] = df['Giá'].str.replace(' tỷ', '', regex=False)
    df['Giá'] = df['Giá'].str.replace(',', '.')
    df['Giá'] = pd.to_numeric(df['Giá'], errors='coerce')

    # Chuyển đổi cột 'Diện tích' từ '72 m2' thành 72.0
    df['Diện tích'] = df['Diện tích'].str.replace(' m2', '', regex=False)
    df['Diện tích'] = pd.to_numeric(df['Diện tích'], errors='coerce')

    # Chuyển đổi các cột 'Chiều ngang', 'Chiều dài', 'Đường trước nhà' từ '5,5m' thành 5.5
    for col in ['Chiều ngang', 'Chiều dài', 'Đường trước nhà']:
        df[col] = df[col].astype(str).str.replace('m', '', regex=False)
        df[col] = df[col].str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Chuyển đổi các cột nhị phân từ 'Có' thành 1, 'Không' hoặc '---' thành 0
    binary_cols = ['Phòng ăn', 'Nhà bếp', 'Sân thượng', 'Chỗ để xe hơi', 'Chính chủ']
    for col in binary_cols:
        df[col] = df[col].map({'Có': 1, 'Không': 0, '---': 0})
        df[col] = df[col].fillna(0).astype(int)

    # Lưu dữ liệu đã xử lý vào file CSV mới
    df.to_csv(output_file, index=False)
    print(f"✅ Dữ liệu đã được chuyển đổi và lưu tại: {output_file}")

# Ví dụ sử dụng
input_path = 'F:/3.5 Years/First Year/Python/Mini_Linear/final_data.csv'  # Đường dẫn đến file đầu vào
output_path = 'F:/3.5 Years/First Year/Python/Mini_Linear/final_data_new.csv'  # Đường dẫn đến file đầu ra
convert_real_estate_data(input_path, output_path)