// static/js/main.js

const els = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    previewArea: document.getElementById('previewArea'),
    previewPlaceholder: document.getElementById('previewPlaceholder'),
    previewImg: document.getElementById('previewImg'),
    resultPanel: document.getElementById('resultPanel'),
    advicePanel: document.getElementById('advicePanel'),
    resultList: document.getElementById('resultList'),
    resultCount: document.getElementById('resultCount'),
    noResultMsg: document.getElementById('noResultMsg'),
    adviceContent: document.getElementById('adviceContent'),
    loading: document.getElementById('loading'),
    imageSize: document.getElementById('imageSize'),
    detectTime: document.getElementById('detectTime'),
    predictTimeDisplay: document.getElementById('predictTimeDisplay'),
    currentModelDisplay: document.getElementById('currentModelDisplay'),
    modelSelect: document.getElementById('modelSelect'),
    switchModelBtn: document.getElementById('switchModelBtn'),
    chatMessages: document.getElementById('chatMessages'),
    questionInput: document.getElementById('questionInput'),
    askBtn: document.getElementById('askBtn')
};

let currentDetections = [];
let classColors = [];
let classAdvice = {};
let currentModel = '';
let selectedDetection = null;
let currentFileBlob = null; // 【新增】缓存当前上传的文件对象，用于切换模型时重测

// 初始化
async function init() {
    try {
        const res = await fetch('/class-info');
        const data = await res.json();
        classColors = data.colors.map(rgb => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`);
        classAdvice = data.advice;
    } catch (err) { console.warn('获取类别信息失败', err); }

    await loadModelList();
    els.askBtn.disabled = false;
}

async function loadModelList() {
    try {
        const res = await fetch('/models');
        const data = await res.json();
        els.modelSelect.innerHTML = '';
        data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name;
            els.modelSelect.appendChild(opt);
        });
        currentModel = data.current;
        els.modelSelect.value = currentModel;
        els.currentModelDisplay.textContent = currentModel;
    } catch (err) { console.error('加载模型列表失败', err); }
}

// 【修改】切换模型逻辑
els.switchModelBtn.addEventListener('click', async () => {
    const newModel = els.modelSelect.value;
    if (newModel === currentModel) return;

    els.loading.classList.remove('hidden');

    try {
        const res = await fetch('/set_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: newModel })
        });
        const data = await res.json();

        if (data.success) {
            currentModel = newModel;
            els.currentModelDisplay.textContent = newModel;

            // 【核心逻辑】如果当前已经有图片（缓存了 fileBlob），则自动重新预测
            if (currentFileBlob) {
                // 稍微延迟一下，让 Loading 显示出来，体验更好
                setTimeout(() => {
                    runPrediction(currentFileBlob);
                }, 100);
            } else {
                els.loading.classList.add('hidden');
                alert(`模型已切换为：${newModel}`);
            }
        } else {
            els.loading.classList.add('hidden');
            alert('切换模型失败');
        }
    } catch (err) {
        els.loading.classList.add('hidden');
        alert('网络错误');
    }
});

// 上传区域事件
els.uploadArea.addEventListener('click', () => els.fileInput.click());
['dragover', 'dragleave', 'drop'].forEach(evt => {
    els.uploadArea.addEventListener(evt, e => {
        e.preventDefault();
        if (evt === 'dragover') els.uploadArea.classList.add('border-green-500', 'bg-green-50');
        else els.uploadArea.classList.remove('border-green-500', 'bg-green-50');
    });
});
els.uploadArea.addEventListener('drop', e => {
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
els.fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return alert('请上传图片');

    // 【新增】缓存文件对象
    currentFileBlob = file;

    const reader = new FileReader();
    reader.onload = e => {
        // 重置界面状态
        els.previewArea.classList.remove('hidden');
        els.previewPlaceholder.classList.add('hidden');
        els.resultPanel.classList.remove('hidden');
        els.advicePanel.classList.remove('hidden');

        els.resultList.innerHTML = '';
        els.noResultMsg.classList.add('hidden');
        els.adviceContent.textContent = '分析中...';
        els.imageSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
        els.detectTime.textContent = '检测中...';

        // 暂时显示原图（可选，提升体验），等预测完成后会被带框图覆盖
        // els.previewImg.src = e.target.result;

        // 执行预测
        runPrediction(file);
    };
    reader.readAsDataURL(file);
}

// 【重构】独立的预测执行函数
function runPrediction(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', currentModel);

    els.loading.classList.remove('hidden');
    const startTime = Date.now();

    fetch('/predict', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            els.loading.classList.add('hidden');
            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            els.detectTime.textContent = `${duration} 秒`;
            if (data.inference_time) els.predictTimeDisplay.textContent = `${data.inference_time} ms`;

            if (data.success) {
                currentDetections = data.detections;
                displayResults(data);
                // 如果是切换模型触发的，可以给个微小提示（可选）
                if (!file) { /* 这里的 file 永远是有的，因为是传进来的 */ }
            } else {
                alert('检测失败：' + data.message);
            }
        })
        .catch(err => {
            els.loading.classList.add('hidden');
            console.error(err);
            alert('网络错误');
        });
}

// 兼容旧的 sendImage 调用（实际上现在统一用 runPrediction）
function sendImage(file) {
    runPrediction(file);
}

function displayResults(data) {
    const detections = data.detections;

    if (data.annotated_image) {
        els.previewImg.src = data.annotated_image;
    }

    els.resultCount.textContent = detections.length ? `${detections.length}个目标` : '无目标';

    if (detections.length === 0) {
        els.noResultMsg.classList.remove('hidden');
        els.adviceContent.textContent = '未检测到病害，植株健康。';
        els.resultList.innerHTML = '';
        return;
    }
    els.noResultMsg.classList.add('hidden');

    let html = '';
    detections.forEach((det, idx) => {
        const score = (det.score * 100).toFixed(1);
        const name = det.class_name || '未知';
        const color = classColors[det.class_id] || 'rgb(255,0,0)';

        html += `
        <div class="result-item p-3 rounded-lg flex justify-between items-center text-sm border-l-4 bg-white shadow-sm"
             style="border-color: ${color}" data-index="${idx}">
            <span class="font-medium text-gray-700">${name}</span>
            <span class="text-xs bg-gray-100 px-2 py-1 rounded text-gray-600">${score}%</span>
        </div>`;
    });
    els.resultList.innerHTML = html;

    document.querySelectorAll('.result-item').forEach(item => {
        item.addEventListener('click', function() {
            document.querySelectorAll('.result-item').forEach(i => {
                i.classList.remove('active', 'bg-green-50');
                i.style.borderLeftWidth = '4px';
            });
            this.classList.add('active', 'bg-green-50');
            this.style.borderLeftWidth = '6px';

            const idx = this.dataset.index;
            selectedDetection = detections[idx];
            els.adviceContent.textContent = selectedDetection.advice || '暂无建议';
        });
    });

    if (detections.length > 0) els.resultList.firstElementChild.click();
}

// 聊天功能
els.askBtn.addEventListener('click', async () => {
    const q = els.questionInput.value.trim();
    if (!q) return;

    addMessage(q, 'user');
    els.questionInput.value = '';

    const typingId = showTyping();

    try {
        const res = await fetch('/ask_qwen', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, disease_info: selectedDetection })
        });
        const data = await res.json();
        removeTyping(typingId);
        if (data.success) addMessage(data.answer, 'assistant');
        else addMessage('AI 服务错误：' + data.message, 'assistant');
    } catch (e) {
        removeTyping(typingId);
        addMessage('网络错误', 'assistant');
    }
});

function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `chat-message text-sm ${sender === 'user' ? 'user-message' : 'assistant-message'}`;
    div.textContent = text;
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function showTyping() {
    const id = 't-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'assistant-message chat-message typing-indicator';
    div.innerHTML = '<span></span><span></span><span></span>';
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

init();