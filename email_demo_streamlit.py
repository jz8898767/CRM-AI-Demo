import streamlit as st
import openai
import streamlit.components.v1 as components
import requests
import numpy as np
import re
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size=300):#切块
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def retrieve_top_chunks_tfidf(chunks, query, top_k=3):# TF-IDF 向量化和余弦相似度进行检索
    vectorizer = TfidfVectorizer()# 初始化 TF-IDF 向量器
    chunk_vectors = vectorizer.fit_transform(chunks)# 将文本块转换为 TF-IDF 向量
    query_vector = vectorizer.transform([query])# 将查询转换为 TF-IDF 向量
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]# 计算查询与每个文本块的余弦相似度
    top_indices = similarities.argsort()[-top_k:][::-1]# 获取相似度最高的 top_k 个文本块的索引
    return [chunks[i] for i in top_indices]

def get_qwen_embedding(text, api_key):# 调用 Qwen Embedding API 
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-v2"
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:# 错误处理
        raise Exception(f"Qwen Embedding API error: {response.status_code} - {response.text}")
    result = response.json()
    return np.array(result['output']['embeddings'][0]['embedding'])

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def retrieve_top_chunks_embedding(chunks, query, api_key, top_k=3):# Qwen Embedding API 向量化和余弦相似度进行检索
    query_vec = get_qwen_embedding(query, api_key)# 获取查询的向量表示
    chunk_vectors = [get_qwen_embedding(chunk, api_key) for chunk in chunks]# 获取每个文本块的向量表示
    similarities = [cosine_sim(query_vec, cv) for cv in chunk_vectors]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Streamlit 页面配置
st.set_page_config(page_title="游戏CRM邮件生成系统", layout="wide")
st.markdown("""
<style>
.mobile-upload-tip {
    text-align: center;
    font-size: 13px;
    color: #666;
    margin: 8px 0;
    display: none;
}
@media (max-width: 768px) {
    .mobile-upload-tip {
        display: block;
    }
}
</style>
<div class="mobile-upload-tip">
📱 移动端用户请点击左上角「☰」，打开侧边栏 选择目标用户 & 上传 RAG 知识库
</div>
""", unsafe_allow_html=True)
st.title("游戏CRM邮件生成系统(Demo)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 系统配置")

    api_key = ""
    try:
        if "api_key" in st.secrets:
            api_key = st.secrets["api_key"]
            st.success("✅ 已从云端安全加载 API 密钥")
        else:
            api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    except:
        api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    
    dashscope_api_key = ""
    try:
        if "dashscope_api_key" in st.secrets:
            dashscope_api_key = st.secrets["dashscope_api_key"]
            st.success("✅ 已加载 DashScope (Qwen) API 密钥")
        else:
            dashscope_api_key = st.text_input("DashScope API Key（可选，用于语义检索；留空则使用关键词匹配）", type="password")
    except:
        dashscope_api_key = st.text_input("DashScope API Key（可选，用于语义检索；留空则使用关键词匹配）", type="password")

    st.markdown("---")
    st.header("👥 目标用户分层（仅展示，尚未完善该功能）")
    user_segment = st.selectbox(
        "选择目标客群",
        ["流失老玩家 (30天未登, 高付费潜力)", "活跃新玩家 (7天新进, 低付费)", "大R核心玩家 (持续活跃, 高客单)"]
    )

    st.markdown("---")
    st.header("上传知识库 (RAG)")

    uploaded_file = st.file_uploader("上传游戏 Wiki 或版本指南 (.txt)", type=("txt"))

    kb_content = ""
    if uploaded_file:
        kb_content = uploaded_file.read().decode("utf-8")
        st.success("✅ 知识库已加载")

        st.info(f"知识库长度：{len(kb_content)} 字符")


col1, col2 = st.columns([1, 1.2])
with col1:
    st.subheader("活动简报输入")
    preset_options = {
        "自定义输入": "",
        "《王者荣耀》S34 赛季“云梦有灵”回归活动": (
            "核心目的：利用新赛季热度，配合高价值福利，唤醒 30 天以上未登录的老玩家。\n"
            "活动权益：\n"
            "1. 回归专属礼包：登录即送“英雄碎片*20” + “排位保护卡*1”。\n"
            "2. 限时挑战：完成 3 局排位，必得“史诗皮肤自选宝箱”。\n"
            "A/B 测试策略要求：\n"
            "- 方案 A（紧迫感）：强调“S34 赛季限定”和“回归福利倒计时”。\n"
            "- 方案 B（情感）：强调“昔日战友在等你”、“峡谷需要你”的情感连接，唤起玩家的归属感。\n"
            "风格要求：神秘、梦幻，深蓝与金色为主色调。"
        ),
        "科幻 FPS 新游《星际战魂》封闭内测预约": (
            "核心目的：邀请高净值（大 R）玩家参与首测，强调尊贵感和特权，转化为核心种子用户。\n"
            "活动权益：\n"
            "1. 绝版称号：“星际先驱者”（公测永久保留，带特效）。\n"
            "2. 充值返利：内测期间充值，公测 200% 返还点券。\n"
            "3. 专属客服：1对1 专属管家服务通道。\n"
            "A/B 测试策略要求：\n"
            "- 方案 A（尊贵感）：侧重“限量名额”和“身份象征”，强调只有顶尖玩家才有资格参与。\n"
            "- 方案 B（利益点）：侧重“200% 高额返利”和“绝版资产”，强调投资回报率和数值优势。\n"
            "风格要求：硬核科幻，赛博朋克，黑金配色，展现高端质感。"
        )
    }

    selected_preset = st.selectbox(
        "快速加载模板：",
        list(preset_options.keys())
    )

    campaign_brief = st.text_area(
        "请在此描述活动内容、目标用户与A/B 测试要求：",
        value=preset_options[selected_preset],
        height=250
    )

    generate_btn = st.button("开始 AI 自动生成", use_container_width=True)

with col2:
    st.subheader("A/B 测试生成与质量评估")

    if generate_btn:
        if not api_key:
            st.error("❌ 请先配置 API Key！")
        else:
            try:
                client = openai.OpenAI(
                    api_key=api_key, 
                    base_url="https://api.deepseek.com"
                )

                with st.spinner("AI 正在生成 A/B 两版方案并进行合规质检..."):
                    
                    format_instruction = """
                    请严格按照以下标记格式输出内容：

                    ===VARIANT_A===
                    (在这里写 A 版 HTML 代码)
                    ===END_A===

                    ===STRATEGY_A===
                    (在这里一句话概括 A 版策略)
                    ===END_STRATEGY_A===

                    ===VARIANT_B===
                    (在这里写 B 版 HTML 代码)
                    ===END_B===

                    ===STRATEGY_B===
                    (在这里一句话概括 B 版策略)
                    ===END_STRATEGY_B===

                    ===SCORE===
                    (在这里只写分数数字，例如：88)
                    ===END_SCORE===

                    ===REASON===
                    (在这里写评分理由)
                    ===END_REASON===
                    """

                    if kb_content:# 如果上传了知识库，启用 RAG 增强模式；否则使用普通生成模式
                        st.success("启用 RAG 检索增强模式")
                        try:
                            chunks = chunk_text(kb_content)
                            try:# 尝试 Embedding 检索，失败回退到 TF-IDF
                                if dashscope_api_key:
                                    top_chunks = retrieve_top_chunks_embedding(chunks, campaign_brief, dashscope_api_key, top_k=3)
                                else:
                                    top_chunks = retrieve_top_chunks_tfidf(chunks, campaign_brief, top_k=3)
                            except:
                                top_chunks = retrieve_top_chunks_tfidf(chunks, campaign_brief, top_k=3)
                            
                            retrieved_context = "\n".join(top_chunks)
                            with st.expander("查看检索到的知识片段"):
                                st.code(retrieved_context)
                        except Exception as e:
                            st.warning(f"检索异常，已降级处理：{e}")
                            retrieved_context = "检索失败"

                        system_prompt = f"""
                        你是一个游戏 CRM 专家。请根据活动简报和参考资料生成 A/B 测试邮件。

                        【目标用户】：{user_segment}
                        【参考资料】：{retrieved_context}

                        要求：
                        - 邮件术语必须与参考资料一致
                        - 风格符合游戏调性
                        - 页脚涉及到运营团队的称谓必须与游戏名称一致
                        - 包含：标题、副标题、正文、CTA按钮、页脚
                        - 使用简洁内联 CSS
                        - CTA 按钮必须是 <a href="https://jz8898767.github.io/egg_page/">
                        - 【目标用户】与【活动简报】中的目标用户冲突时，以【活动简报】为准

                        {format_instruction}
                        """

                    else:# 普通生成模式 (无知识库)
                        st.info("未上传知识库，使用通用模型生成")

                        system_prompt = f"""
                        你是一个游戏 CRM 专家。请根据活动简报生成 A/B 测试邮件。
                        
                        【目标用户】：{user_segment}

                        要求：
                        - 邮件术语必须与参考资料一致
                        - 风格符合游戏调性
                        - 页脚涉及到运营团队的称谓必须与游戏名称一致
                        - 包含：标题、副标题、正文、CTA按钮、页脚
                        - 使用简洁内联 CSS
                        - CTA 按钮必须是 <a href="https://jz8898767.github.io/egg_page/">
                        - 【目标用户】与【活动简报】中的目标用户冲突时，以【活动简报】为准

                        {format_instruction}
                        """

                    # 统一调用 API 
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "user", "content": system_prompt + f"\n\n【活动简报】：{campaign_brief}"}
                        ],
                        temperature=0.7
                    )

                    raw_content = response.choices[0].message.content

                    # 统一解析
                    def safe_extract(text, start_tag, end_tag, default_val):# 使用正则表达式提取内容，避免标签缺失导致的错误
                        pattern = f"{start_tag}(.*?){end_tag}"# re.DOTALL 让 . 匹配换行符，确保能提取多行内容
                        match = re.search(pattern, text, re.DOTALL)# 如果匹配成功，返回提取的内容；否则返回默认值
                        if match:
                            return match.group(1).strip()
                        return default_val

                    html_a = safe_extract(raw_content, "===VARIANT_A===", "===END_A===", "<div>A版生成失败</div>")
                    strat_a = safe_extract(raw_content, "===STRATEGY_A===", "===END_STRATEGY_A===", "通用策略")
                    
                    html_b = safe_extract(raw_content, "===VARIANT_B===", "===END_B===", "<div>B版生成失败</div>")
                    strat_b = safe_extract(raw_content, "===STRATEGY_B===", "===END_STRATEGY_B===", "通用策略")
                    
                    score = safe_extract(raw_content, "===SCORE===", "===END_SCORE===", "0")
                    reason = safe_extract(raw_content, "===REASON===", "===END_REASON===", "AI 未能生成评价")

                    #  UI 展示
                    st.info(f"**AI 质量合规评分（仅展示，尚未完善该功能）：{score}/100**")
                    st.caption(f"评审意见（仅展示，尚未完善该功能）：{reason}")
                    
                    st.divider()

                    tab_a, tab_b = st.tabs(["方案 A", "方案 B"])
                    
                    with tab_a:
                        st.write(f"**策略思路**：{strat_a}")
                        components.html(html_a, height=600, scrolling=True)
                        st.download_button("💾 下载方案 A", html_a, "email_variant_a.html")
                        with st.expander("查看源代码"):
                            st.code(html_a, language="html")
                        
                    with tab_b:
                        st.write(f"**策略思路**：{strat_b}")
                        components.html(html_b, height=600, scrolling=True)
                        st.download_button("💾 下载方案 B", html_b, "email_variant_b.html")
                        with st.expander("查看源代码"):
                            st.code(html_b, language="html")

            except Exception as e:
                st.error(f"运行出错：{str(e)}")
