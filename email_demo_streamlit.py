import streamlit as st
import openai
import streamlit.components.v1 as components

# 1. 页面配置：设置网页标题和布局
st.set_page_config(page_title="腾讯游戏 CRM 智能生成平台", layout="wide")

st.title("🎮 腾讯游戏 CRM 智能邮件生成系统")
st.markdown("---")

# 2. 侧边栏配置：填入 API 信息
with st.sidebar:
    st.header("⚙️ 系统配置")
    if "api_key" in st.secrets:
        api_key = st.secrets["api_key"]
        st.success("✅ API 密钥已从系统配置中加载")
    else:
        api_key = st.text_input("请输入 DeepSeek API Key", type="password")
        st.info("提示：部署时可在 Advanced Settings 中配置 Secrets 以实现免输入。")
    model_choice = st.selectbox("选择模型", ["deepseek-chat"])
    st.info("本原型用于演示：输入简报 → 自动生成品牌对齐的 HTML 邮件")
    
# 3. 主界面布局：左侧输入，右侧输出
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 活动简报输入 (Ingestion)")
    preset_options = {
        "自定义输入": "",
        "🔥 腾讯游戏：赛博春季回归活动": "项目：腾讯自研新游《星际战魂》赛博春季赛回归活动。\n目标人群：30天未登录的高价值流失玩家。\n核心奖励：限定传说级皮肤‘极光之刃’限时 8 折，登录即领‘暖春礼包’。\n视觉风格：赛博朋克深色主题，霓虹紫与亮金配色。",
        "🎁 腾讯游戏：新赛季预热活动": "项目：王者荣耀新赛季。目标：吸引活跃玩家参与预注册。奖励：抢先体验卡。风格：竞技感、明亮蓝白配色。"
    }
    selected_preset = st.selectbox("💡 快速加载预设模板：", list(preset_options.keys()))
    campaign_brief = st.text_area(
        "请描述游戏活动内容：",
        value=preset_options[selected_preset],
        height=250
    )
    generate_btn = st.button("🚀 开始 AI 自动化生成", use_container_width=True)

with col2:
    st.subheader("📤 AI 邮件预览 (Output)")
    
    if generate_btn:
        if not api_key:
            st.error("请先在左侧填入 API Key！")
        elif not campaign_brief:
            st.warning("请先输入活动简报内容！")
        else:
            try:
                # 4. 调用 AI 生成模块
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                
                with st.spinner("AI 正在构思精美邮件并渲染 HTML..."):
                    prompt = f'''你是一名资深游戏 CRM 运营。请根据以下简报生成一封生产级别的 HTML 邮件：
                    \n{campaign_brief}\n\n
                    要求:
                    - 仅输出HTML.
                    - 包含: 标题，副标题，邮件正文，CTA按钮，页脚。
                    - 使用简洁的内联 CSS。
                    - CTA 按钮必须是一个带样式的 <a> 标签。
                    - 语调： 简洁、友好、值得信赖，符合该游戏一贯的语调和用词。
                    - 品牌指南： 游戏风格、字体和元素使用该游戏中最常出现的颜色、高能量的视觉布局。
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
                    </html>'''
                    
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )
                    html_content = response.choices[0].message.content
                    
                    # 5. 实时渲染预览
                    st.success("邮件生成成功！")
                    components.html(html_content, height=500, scrolling=True)
                    
                    # 提供下载功能，符合“部署与交付”目标
                    st.download_button(
                        label="💾 下载生成的 HTML 文件",
                        data=html_content,
                        file_name="tencent_crm_email.html",
                        mime="text/html"
                    )
            except Exception as e:

                st.error(f"生成失败：{str(e)}")

