import joblib
import numpy as np
import pandas as pd

def predict_from_input(model):
    # Nhập các thông tin đầu vào từ người dùng
    print("Nhập thông tin bất động sản cần dự đoán:")
    area = float(input("Diện tích (m2): "))
    width = float(input("Chiều ngang (m): "))
    length = float(input("Chiều dài (m): "))
    street_width = float(input("Đường trước nhà (m): "))
    floors = int(input("Số lầu: "))
    bedrooms = int(input("Số phòng ngủ: "))

    # Nhập các thông tin nhị phân
    dining = input("Có phòng ăn? (y/n): ").lower() == 'y'
    kitchen = input("Có nhà bếp? (y/n): ").lower() == 'y'
    terrace = input("Có sân thượng? (y/n): ").lower() == 'y'
    car_park = input("Có chỗ để xe hơi? (y/n): ").lower() == 'y'
    owner = input("Chính chủ? (y/n): ").lower() == 'y'

    # Nhập thông tin phân loại
    street = input("Tên đường: ")
    ward = input("Phường: ")
    district = input("Quận: ")
    city = input("Thành phố: ")
    property_type = input("Loại BDS: ")
    direction = input("Hướng: ")

    # Tạo DataFrame chứa dữ liệu nhập vào
    new_data = pd.DataFrame([{
        'Diện tích': area,
        'Chiều ngang': width,
        'Chiều dài': length,
        'Đường trước nhà': street_width,
        'Số lầu': floors,
        'Số phòng ngủ': bedrooms,
        'Phòng ăn': int(dining),
        'Nhà bếp': int(kitchen),
        'Sân thượng': int(terrace),
        'Chỗ để xe hơi': int(car_park),
        'Chính chủ': int(owner),
        'Đường': street,
        'Phường': ward,
        'Quận': district,
        'Thành phố': city,
        'Loại BDS': property_type,
        'Hướng': direction
    }])

    # Dự đoán
    log_price = model.predict(new_data)[0]
    price = np.expm1(log_price)

    print(f"💰 Giá bất động sản dự đoán: khoảng {price:.2f} tỷ đồng")


loaded_model = joblib.load('F:/3.5 Years/First Year/Python/Mini_Linear/trained_rf_model.pkl')

# Gọi hàm dự đoán từ input người dùng
predict_from_input(loaded_model)