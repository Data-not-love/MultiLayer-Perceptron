import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib



# Hàm huấn luyện mô hình MLP
def train_mlp_model(data):
    data.rename(columns={
        'Giá (Tỷ)': 'Giá',
        'Diện tích (M2)': 'Diện tích'
    }, inplace=True)

    # Danh sách cột số
    numeric_features = ['Diện tích', 'Chiều ngang', 'Chiều dài', 'Đường trước nhà', 'Số lầu', 'Số phòng ngủ']
    # Cột nhị phân
    binary_cols = ['Phòng ăn', 'Nhà bếp', 'Sân thượng', 'Chỗ để xe hơi', 'Chính chủ']
    # Cột phân loại
    categorical_features = ['Đường', 'Phường', 'Quận', 'Thành phố', 'Loại BDS', 'Hướng']

    # Ép kiểu numeric và kiểm tra lỗi
    for col in numeric_features + binary_cols + ['Giá']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Xử lý giá trị thiếu ở cột 'Giá'
    data = data.dropna(subset=['Giá'])

    # Điền giá trị thiếu cho phân loại
    data[categorical_features] = data[categorical_features].fillna('Unknown')


    # Tạo đặc trưng X và mục tiêu y
    X = data[numeric_features + categorical_features + binary_cols]
    y = np.log1p(data['Giá'])  # log(1 + Giá)

    # Tách tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')), # thay các giá trị Nan về Median
                ('scaler', StandardScaler()) #Chuẩn hóa dữ liệu về dạng có trung bình = 0, độ lệch chuẩn = 1 → Mục tiêu là giúp mô hình hội tụ tốt hơn khi dữ liệu đầu vào đồng đều.
            ]), numeric_features + binary_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), # Nan replace with unkonw
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # chưa có giá trị thì b qun thay vì báo lỗi
            ]), categorical_features)
        ])


#multilayer perceptron
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=1500,
            random_state=42
            #early_stopping=True,
            #validation_fraction=0.1
        ))
    ])

    # Huấn luyện mô hình
    model_pipeline.fit(X_train, y_train)

    # Dự đoán và chuyển lại dạng giá gốc
    log_predictions = model_pipeline.predict(X_test)
    predictions = np.expm1(log_predictions)

    # Đánh giá mô hình
    y_test_original = np.expm1(y_test)
    mse = mean_squared_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)



    print("📊 Một số dự đoán mẫu:")
    results_df = pd.DataFrame({
        'Giá thực tế (Tỷ)': y_test_original.values,
        'Giá dự đoán (Tỷ)': predictions
    })
    print(results_df.head(100).to_string(index=False))


    print(f"\n🧠 Mean Squared Error (MSE): {mse:.2f}")
    print(f"🧠 R-squared (R²): {r2:.2f}\n")
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_original, predictions, alpha=0.7)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel("Giá thực tế (Tỷ)")
    plt.ylabel("Giá dự đoán (Tỷ)")
    plt.title("So sánh Giá thực tế vs Dự đoán (MLP)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model_pipeline


# Đọc file dữ liệu
data = pd.read_csv('F:/3.5 Years/First Year/Python/Mini_Linear/final_data_clean_1.csv')

# Huấn luyện mô hình
trained_model = train_mlp_model(data)
joblib.dump(trained_model, 'F:/3.5 Years/First Year/Python/Mini_Linear/trained_rf_model.pkl')
print("💾 Mô hình đã được lưu vào 'trained_rf_model.pkl'")