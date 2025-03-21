import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from model import EndToEndHierarchicalModel  # Import model from your code (make sure it's in the same directory or adjust accordingly)
import torch.nn.functional as F
import requests
from io import BytesIO
import pickle

# Tải ánh xạ vào Flask
with open("mappings.pkl", "rb") as f:
    all_mappings = pickle.load(f)

# Lấy từng ánh xạ từ dictionary
inv_brand_map = all_mappings['inv_brand_map']
inv_gender_map = all_mappings['inv_gender_map']
inv_usage_map = all_mappings['inv_usage_map']
inv_base_colour_map = all_mappings['inv_base_colour_map']
inv_master_map = all_mappings['inv_master_map']
inv_sub_cat_map = all_mappings['inv_sub_cat_map']
inv_article_type_map = all_mappings['inv_article_type_map']

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


# Route cho trang chủ (Home)
@app.route('/')
def home():
    return 'Welcome to the Product Prediction API! Please use the /predict endpoint to make predictions.'

# Load mô hình AI đã train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EndToEndHierarchicalModel(
    num_gender=5, num_master=7, num_usage=8,
    num_sub=45, num_article=141, num_base=46,
    num_brand=817,
    pretrained=True,
    mlp_hidden=32,
    embed_dim=16,
    teacher_forcing_p=0.5
).to(device)

model.load_state_dict(torch.load(r"C:\Users\Admin\OneDrive - uel.edu.vn\Documents\GitHub\web2\blockchain-scm\models\CNN_hierarchical_model.pt", map_location=torch.device('cpu')))

model.eval()


transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dự đoán tên sản phẩm dựa trên URL ảnh và dữ liệu numeric
def predict_product_display_name(image_url, numeric_data=None):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_tensor = transform_infer(img).unsqueeze(0)

    if numeric_data is not None:
        numeric_data = torch.tensor(numeric_data, dtype=torch.float).unsqueeze(0)
    else:
        numeric_data = None

    with torch.no_grad():
        outputs = model(img_tensor.to(device), numeric_data=numeric_data.to(device) if numeric_data is not None else None)

    # Dự đoán các nhãn
    pred_brand_id = torch.argmax(outputs["brand"], dim=1).item()
    pred_gender_id = torch.argmax(outputs["gender"], dim=1).item()
    pred_usage_id = torch.argmax(outputs["usage"], dim=1).item()
    pred_base_id = torch.argmax(outputs["baseColour"], dim=1).item()
    pred_masterCategory_id = torch.argmax(outputs["masterCategory"], dim=1).item()
    pred_subCategory_id = torch.argmax(outputs["subCategory"], dim=1).item()
    pred_article_id = torch.argmax(outputs["articleType"], dim=1).item()

    # Chuyển ID thành chuỗi nhãn
    pred_brand_str = inv_brand_map[pred_brand_id]
    pred_gender_str = inv_gender_map[pred_gender_id]
    pred_usage_str = inv_usage_map[pred_usage_id]
    pred_base_str = inv_base_colour_map[pred_base_id]
    pred_masterCategory_str = inv_master_map[pred_masterCategory_id]
    pred_subCategory_str = inv_sub_cat_map[pred_subCategory_id]
    pred_article_str = inv_article_type_map[pred_article_id]

    # Ghép "product display name"
    predicted_display_name = f"{pred_brand_str} {pred_gender_str} {pred_usage_str} {pred_base_str} {pred_masterCategory_str} {pred_subCategory_str} {pred_article_str}"

    # Trả về 7 nhãn và tên sản phẩm hiển thị
    return {
        "brand": pred_brand_str,
        "gender": pred_gender_str,
        "usage": pred_usage_str,
        "baseColour": pred_base_str,
        "masterCategory": pred_masterCategory_str,
        "subCategory": pred_subCategory_str,
        "articleType": pred_article_str,
        "predicted_display_name": predicted_display_name
    }



# Route để nhận ảnh và trả kết quả
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Trả về 200 OK cho yêu cầu OPTIONS (preflight)
        return '', 200
    data = request.get_json()
    # Nhận URL của ảnh từ request
    image_url = data.get('image_url')
    print(image_url)
    if not image_url:
        return jsonify({"error": "No image URL provided"})
    

    # Dự đoán
    prediction = predict_product_display_name(image_url)

    # Trả kết quả về cho người dùng
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
