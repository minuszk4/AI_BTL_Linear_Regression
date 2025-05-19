from flask import Flask, jsonify
import json
from flask_cors import CORS
from  model_code import Linear_Regression
from flask import request
import numpy as np
model = Linear_Regression()
model.load_model('./models/linear_regression_model.pkl')
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

with open("./data/district_ward_list.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    # print(raw_data)
with open("./data/investors.json", "r", encoding="utf-8") as f:
    investors = json.load(f)

districts = [
    {
        "district": d["district"],
        "ward": [w for w in d["ward"] ]
    }
    for d in raw_data
]
@app.route("/api/investors", methods=["GET"])
def get_investors():
    return jsonify(investors)
@app.route("/api/districts", methods=["GET"])
def get_districts():
    return jsonify([d["district"] for d in districts])

@app.route("/api/wards/<district_name>", methods=["GET"])
def get_wards(district_name):
    for d in districts:
        if d["district"].lower() == district_name.lower():
            return jsonify(d["ward"])
    return jsonify([]), 404


def one_hot_encode(features, categories, columns, drop_first=True):
    encoded_features = []
    categorical_columns = ['investor','direction', 'balcony', 'district', 'ward']
    
    feature_names = []
    
    for idx, col in enumerate(columns):
        if col in categorical_columns and col in categories:
            print(f"Encoding column: {col}")
            unique_vals = categories[col]
            print(f"Unique values: {unique_vals}")
            col_val = features[idx]
            if col_val not in unique_vals:
                raise ValueError(f"Value {col_val} not in {col} categories")
            vals_to_encode = unique_vals[1:] if drop_first else unique_vals
            encoded_values = [1 if val == col_val else 0 for val in vals_to_encode]
            encoded_features.extend(encoded_values)
            feature_names.extend([f"{col}_{val}" for val in vals_to_encode])
        else:
            encoded_features.append(features[idx])
            feature_names.append(col)
    
    print(f"Feature names after encoding: {feature_names}")
    return encoded_features

def standardize_transform(X, mean, std):
    X = np.array(X, dtype=np.float64) 
    if mean is None or std is None:
        raise ValueError("Mean or std is None. Ensure the model was loaded correctly.")
    if X.shape[1] != len(mean):
        raise ValueError(f"Feature count mismatch: got {X.shape[1]}, expected {len(mean)}")
    if np.any(np.isnan(X)) or np.any(np.isnan(mean)) or np.any(np.isnan(std)):
        raise ValueError("NaN values detected in input, mean, or std")
    std = np.where(std == 0, 1, std)
    return (X - mean) / std

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['investor','squares', 'bedrooms', 'bathrooms', 'direction', 'balcony', 'district', 'ward']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

        try:
            investor = str(data['investor'])
            squares = float(data['squares'])
            bedrooms = int(data['bedrooms'])
            bathrooms = int(data['bathrooms'])
            direction = str(data['direction']).strip()
            balcony = str(data['balcony']).strip()
            district = str(data['district']).strip()
            ward = str(data['ward']).strip()
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid data types: {str(e)}'}), 400

        print(f"Input: investor ={investor} squares={squares}, bedrooms={bedrooms}, bathrooms={bathrooms}, "
              f"direction={direction}, balcony={balcony}, district={district}, ward={ward}")

        categorical_columns = ['investor', 'direction', 'balcony', 'district', 'ward']
        categories = {
        'direction': ['Đông-Nam', 'unknown', 'Tây-Bắc', 'Tây-Nam', 'Bắc', 'Đông-Bắc', 'Nam', 'Đông', 'Tây'],
        'balcony': ['Tây-Bắc', 'none', 'Đông-Nam', 'Đông-Bắc', 'Đông', 'Nam', 'Tây', 'Tây-Nam', 'Bắc'],
        'district': [d['district'] for d in districts],  # Đảm bảo `districts` là list chứa dict có key 'district'
        'ward': [
            'dai mo', 'phu dien', 'tu hiep', 'trung hoa', 'hoang liet', 'me tri',
            'phuc loi', 'unknown', 'nguyen trai', 'kim chung', 'co nhue 1',
            'thanh xuan trung', 'phuc la', 'minh khai', 'yen hoa', 'duong xa', 'my dinh 1',
            'quang trung', 'an khanh', 'mo lao', 'dong ngac', 'xuan la', 'duong noi',
            'cau dien', 'dai kim', 'la khe', 'ha cau', 'tan trieu', 'trung van', 'my dinh',
            'phu thuong', 'kim ma', 'nghia do', 'nghia tan', 'vinh tuy', 'thinh liet',
            'thanh xuan nam', 'sai dong', 'bo de', 'thuong dinh', 'gia thuy', 'dong hoi',
            'ha dinh', 'viet hung', 'giang bien', 'yen so', 'my dinh 2', 'nhan chinh',
            'thach hoa', 'dich vong', 'da ton', 'mai dich', 'quan hoa', 'phu la',
            'thuy khue', 'khuong mai', 'xuan dinh', 'tay mo', 'tan lap', 'co nhue 2',
            'yen nghia', 'van quan', 'kien hung', 'phuong canh', 'cong vi', 'ngoc lam',
            'thanh xuan bac', 'phuc dong', 'dang xa', 'o cho dua', 'long bien',
            'lang thuong', 'tran phu', 'vinh ngoc', 'tan mai', 'vinh hung', 'trung liet',
            'buoi', 'giap bat', 'tho quan', 'khuong dinh', 'dinh cong', 'dong tam',
            'phu luong', 'bach mai', 'phuong liet', 'lang ha', 'nhat tan', 'xuan tao',
            'vinh phuc', 'khuong trung', 'thanh cong', 'mai dong', 'xuan phuong',
            'khuong thuong', 'thanh nhan', 'phuong mai', 'thuong thanh', 'duc giang',
            'hang bot', 'doi can', 'duc thuong', 'giang vo', 'van phuc', 'tuong mai',
            'chuc son', 'kim lien', 'quynh mai', 'bach dang', 'le dai hanh', 'lieu giai',
            'lien ninh', 'pham dinh ho', 'kim giang', 'cua nam', 'thinh quang',
            'nga tu so', 'dong mac', 'hang bai', 'phan chu trinh', 'thanh luong',
            'dong nhan', 'trung tu', 'truong dinh', 'ngu hiep', 'ngoc khanh', 'thach ban',
            'co nhue', 'quynh loi', 'hoang van thu', 'linh nam', 'van chuong', 'duyen ha',
            'ngoc ha', 'kham thien', 'tran hung dao', 'tay tuu', 'bach khoa', 'thanh tri',
            'phuc dien', 'ngoc hoi', 'yet kieu', 'phuc xa', 'trau quy', 'quang an',
            'huu hoa', 'nam dong', 'cat linh', 'co bi', 'cua dong', 'ngoc thuy', 'van canh',
            'dong du', 'dong anh', 'ly thai to', 'phuong lien', 'hang buom', 'pho hue',
            'cu khe', 'dong xuan', 'trang tien', 'chuong duong', 'kieu ky', 'la phu',
            'hai boi', 'thuy phuong', 'truc bach', 'hang ma', 'quoc tu giam', 'sai son',
            'quan thanh', 'quang minh', 'dai thinh', 'tien phong', 'hang bo', 'duyen thai',
            'van mieu', 'nguyen trung truc', 'phu do', 'nguyen du', 'dong la', 'cau den',
            'trung phung'
        ],
        'investor': investors
        }
        # print(f"Categories: {categories['investor']}")
        for col in categorical_columns:
            if data[col] not in categories[col]:
                print(f"Invalid {col}: {data[col]} not in {categories[col]}")
                return jsonify({'error': f"Invalid {col}: {data[col]} not in {categories[col]}"}, 400)

        feature_columns = ['squares', 'bedrooms', 'bathrooms','investor', 'direction', 'balcony', 'district', 'ward']
        features = [ squares, bedrooms, bathrooms, investor, direction, balcony, district, ward]
        # print(f"Raw features: {features}")
        encoded_features = one_hot_encode(features, categories, feature_columns, drop_first=False)
        print(f"Encoded features length: {len(encoded_features)}")
        print(f"Expected features: {len(model.w)}")

        if len(encoded_features) != len(model.w):
            return jsonify({
                'error': f'Feature count mismatch: got {len(encoded_features)}, expected {len(model.w)}'
            }), 400
        # print(f"Encoded features: {encoded_features}")
        features_standardized = standardize_transform([encoded_features], model.mean, model.std)
        print(f"Standardized features shape: {features_standardized.shape}")

        predicted_price = model.predict(features_standardized)
        print(f"Predicted price: {predicted_price}")

        return jsonify({'predicted_price': float(predicted_price[0])})

    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
