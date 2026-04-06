from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import time
import uuid
from openai import OpenAI
from utils.inference import predict_with_image, get_class_info

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-prod'
CORS(app)

# 通义千问配置
QWEN_API_KEY = "sk-7eccff53deb14d43b4fa79eb5d8030bf"
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

UPLOAD_FOLDER = 'static/uploads'
MODELS_FOLDER = [r'D:\rice-disease-det\models\yolo', r'D:\rice-disease-det\models\cnn']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
for folder in MODELS_FOLDER:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

AVAILABLE_MODELS = []


def scan_models():
    global AVAILABLE_MODELS
    models = []
    for folder in MODELS_FOLDER:
        if not os.path.exists(folder): continue
        folder_lower = folder.lower()
        model_type = 'yolo' if 'yolo' in folder_lower else ('cnn' if 'cnn' in folder_lower else 'unknown')

        for f in os.listdir(folder):
            if f.endswith('.pt'):
                models.append({
                    'name': f.replace('.pt', ''),
                    'path': os.path.join(folder, f),
                    'type': model_type
                })
    AVAILABLE_MODELS = models
    return models


scan_models()
DEFAULT_MODEL = AVAILABLE_MODELS[0]['name'] if AVAILABLE_MODELS else None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    models = scan_models()
    return render_template('index.html', models=models, default_model=DEFAULT_MODEL)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件上传'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': '无效文件'}), 400

    model_name = request.form.get('model', session.get('current_model', DEFAULT_MODEL))
    session['current_model'] = model_name

    model_info = next((m for m in AVAILABLE_MODELS if m['name'] == model_name), None)
    if not model_info:
        return jsonify({'success': False, 'message': '模型不存在'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    start_time = time.time()
    result = predict_with_image(filepath, model_info['path'], model_info['type'])
    result['inference_time'] = round((time.time() - start_time) * 1000, 2)

    return jsonify(result)


@app.route('/ask_qwen', methods=['POST'])
def ask_qwen():
    data = request.json
    question = data.get('question', '')
    disease_info = data.get('disease_info')

    system_content = "你是一位专业的水稻病害专家。"
    if disease_info:
        user_content = f"""基于以下检测结果回答：
病害：{disease_info.get('class_name')}
置信度：{disease_info.get('score', 0) * 100:.1f}%
问题：{question}
请提供防治建议。"""
    else:
        user_content = f"问题：{question}"

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return jsonify({'success': True, 'answer': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/class-info', methods=['GET'])
def class_info():
    return jsonify(get_class_info())


@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({'models': scan_models(), 'current': session.get('current_model', DEFAULT_MODEL)})


@app.route('/set_model', methods=['POST'])
def set_model():
    data = request.json
    model_name = data.get('model')
    if any(m['name'] == model_name for m in AVAILABLE_MODELS):
        session['current_model'] = model_name
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '模型不存在'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)