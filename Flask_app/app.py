from flask import Flask, render_template, request
import os
from fastai.vision.all import * 
from fastai.vision import *
UPLOAD_FOLDER = os.path.join('static', 'img')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # Lưu file vào thư mục tạm thời
    file_path = 'static/img/' + file.filename
    file.save(file_path)
    new_model = load_learner('intel_scene_classifier.pkl')
    result = new_model.predict(
    item='static/img/' + file.filename
    )
    folder_path = f'static/img/{result[0].replace(" ", "_")}/'
    new_file_path = os.path.join(folder_path, file.filename)
    os.rename(file_path, new_file_path)
    return render_template('result.html', filename=file.filename, result=result[0])

@app.route('/categories')
def show_categories():
    categories = ['Hợp Tác', 'Sản Phẩm Đầu Tư', 'Sản Phẩm Hợp Tác - Dự Án Sách', 'Sự Kiện Quốc Tế', 'Tổ Chức Sự Kiện Kết Nối Kinh Doanh', 'Sản Phẩm Websites', 'Hoạt Động Trong Nước', 'Hoạt Động Quốc Tế']  # Danh sách các danh mục (có thể lấy từ cơ sở dữ liệu, file, ...)

    images_dict = {}
    for category in categories:
            category = category.replace(" ", "_")  
            category_path = os.path.join('static', 'img', category)  # Tạo đường dẫn cho thư mục category      
            images = os.listdir(category_path)  # Đọc danh sách tệp tin trong thư mục
            images_dict[category] = images

    return render_template('categories.html', categories=categories, images=images_dict)

if __name__ == '__main__':
    app.run(debug=True)

