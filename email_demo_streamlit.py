import streamlit as st
import openai
import streamlit.components.v1 as components

# 1. 页面配置与标题
st.set_page_config(page_title="腾讯游戏 CRM 智能生成系统", layout="wide")
st.title("🎮 腾讯游戏 CRM 智能邮件生成系统")
st.markdown("---")

# 2. 侧边栏：安全加载 API 与 RAG 知识库
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

    st.markdown("---")
    st.header("📚 游戏知识库 (RAG)")
    
    uploaded_file = st.file_uploader("上传游戏 Wiki 或版本指南 (.txt)", type=("txt"))
    kb_content = ""
    if uploaded_file:
        kb_content = uploaded_file.read().decode("utf-8")
        st.success("✅ 知识库内容已挂载")

# 3. 主界面布局
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 活动简报输入 (Ingestion)")
    
    preset_options = {
        "自定义输入": "",
        "🔥 王者荣耀：S34 赛季回归活动": (
            "项目：《王者荣耀》S34 赛季回归活动。\n目标：针对 30 天未活跃老玩家进行唤醒。\n"
            "权益：登录领‘传说皮肤体验券’。风格：国风暗金主题，深色背景。"
        ),
        "🎁 腾讯新游：赛博春季预热": (
            "项目：新游《星际战魂》预约。卖点：限定传说皮肤 8 折。\n"
            "风格：赛博朋克深黑主题，霓虹紫高亮配色。"
        )
    }
    selected_preset = st.selectbox("💡 快速加载行业最佳实践模板：", list(preset_options.keys()))
    
    campaign_brief = st.text_area(
        "请在此描述活动内容：",
        value=preset_options[selected_preset],
        height=250
    )
    generate_btn = st.button("🚀 开始 AI 自动化生成", use_container_width=True)

with col2:
    st.subheader("📤 AI 邮件预览 (Output)")
    if generate_btn:
        if not api_key:
            st.error("请先配置 API Key！")
        else:
            try:
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                with st.spinner("正在结合知识库生成精准 HTML 内容..."):
                    
                    rag_context = f"\n【参考知识库内容】:\n{kb_content}" if kb_content else ""
                    
                    prompt = f"""
                    你是一名资深游戏 CRM 运营专家。请根据【简报】并参考【知识库】生成生产级的 HTML 邮件。
                    【简报】:\n{campaign_brief}\n{rag_context}
                    要求:
                    - 仅输出HTML.
                    - 包含: 标题，副标题，邮件正文，CTA按钮，页脚。
                    - 使用简洁的内联 CSS。
                    - CTA 按钮必须是一个带样式的 <a> 标签。
                    - 术语需与知识库一致.
                    - 语调： 简洁、友好、值得信赖。
                    - 品牌指南：字体、颜色、背景和元素应结合游戏本身特色，采用高能量的视觉布局。
                    - 结构:
                    <html>
                        <body>
                        <table> (full email layout)
                            <tr><td>[Headline]</td></tr>
                            <tr><td>[Subheadline]</td></tr>
                            <tr><td>[Body]</td></tr>
                            <tr><td>[CTA Button]</td></tr>
                            <tr><td>[Footer]</td></tr>
                        </table>
                        </body>
                    </html>
                    """
                    
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )
                    # 【修复重点】：使用 .content 避免 TypeError
                    html_content = response.choices[0].message.content
                    
                    components.html(html_content, height=600, scrolling=True)
                    st.download_button("💾 下载 HTML 文件", data=html_content, file_name="game_crm_email.html")
            except Exception as e:
                st.error(f"生成失败：{str(e)}")
