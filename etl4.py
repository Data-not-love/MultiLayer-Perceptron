import pandas as pd

def update_missing_dimensions(input_csv, output_csv):

    #Hàm cập nhật các giá trị thiếu trong cột 'Chiều dài' và 'Chiều ngang' bằng giá trị trung vị của mỗi cột.


    df = pd.read_csv(input_csv)

    # Danh sách các cột cần xử lý
    columns_to_update = ['Chiều dài', 'Chiều ngang']

    for col in columns_to_update:
        if col in df.columns:
            # Chuyển đổi dữ liệu sang kiểu số, nếu chưa
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Tính giá trị trung vị (median) của cột, bỏ qua các giá trị NaN
            median_value = df[col].median()

            # Điền các giá trị thiếu bằng giá trị trung vị
            df[col].fillna(median_value, inplace=True)
        else:
            print(f"Cột '{col}' không tồn tại trong dữ liệu.")

    # Lưu dữ liệu đã cập nhật vào file CSV mới
    df.to_csv(output_csv, index=False)
    print(f"Đã cập nhật và lưu dữ liệu vào '{output_csv}'.")

# Ví dụ sử dụng hàm:
update_missing_dimensions('F:/3.5 Years/First Year/Python/Mini_Linear/final_data_new.csv', 'F:/3.5 Years/First Year/Python/Mini_Linear/final_data_new_1.csv')