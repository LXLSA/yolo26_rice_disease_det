# test_qwen.py
from openai import OpenAI
import os

# ===== 请在这里填写你的通义千问 API Key =====
API_KEY = "sk-*********************"  # <--- 替换成你的真实 Key
# =========================================

# 初始化客户端（指向阿里云的接口地址）
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def test_qwen():
    """测试通义千问API连接"""
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",  # 也可以使用 qwen-turbo，qwen-plus 性能更好
            messages=[
                {"role": "system", "content": "你是一位专业的水稻病害专家。"},
                {"role": "user", "content": "水稻稻瘟病有哪些主要症状？"}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = completion.choices[0].message.content
        print("✅ API 调用成功！回答如下：\n")
        print(answer)
        return True
    except Exception as e:
        print(f"❌ API 调用失败：{e}")
        return False


if __name__ == "__main__":
    test_qwen()
