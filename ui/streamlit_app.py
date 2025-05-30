import streamlit as st
import sys
import os

# Add parent directory to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tutor_agent import EnglishTutorAgent

def main():
    st.set_page_config(
        page_title="English AI Tutor",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 English AI Tutor")
    st.write("Trợ lý AI giúp bạn học tiếng Anh")
    
    # Initialize agent in session state
    if 'tutor' not in st.session_state:
        st.session_state.tutor = EnglishTutorAgent()
        st.session_state.messages = []
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        new_level = st.selectbox(
            "Trình độ của bạn:",
            ['beginner', 'intermediate', 'advanced'],
            index=['beginner', 'intermediate', 'advanced'].index(
                st.session_state.tutor.user_profile['level']
            )
        )
        
        if st.button("Cập nhật trình độ"):
            st.session_state.tutor.update_user_level(new_level)
            st.success(f"Đã cập nhật trình độ: {new_level}")
        
        st.divider()
        
        if st.button("🗑️ Xóa cuộc trò chuyện"):
            st.session_state.tutor = EnglishTutorAgent()
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    st.header("💬 Trò chuyện")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                response = st.session_state.tutor.chat(prompt)
                st.write(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Text analysis section
    st.divider()
    st.header("📝 Phân tích văn bản")
    
    text_to_analyze = st.text_area(
        "Nhập đoạn văn bản tiếng Anh để phân tích:",
        height=100,
        placeholder="Ví dụ: I am study English very hard everyday."
    )
    
    if st.button("🔍 Phân tích"):
        if text_to_analyze.strip():
            with st.spinner("Đang phân tích..."):
                analysis = st.session_state.tutor.analyze_text(text_to_analyze)
                st.write("**Kết quả phân tích:**")
                st.write(analysis)
        else:
            st.warning("Vui lòng nhập văn bản cần phân tích.")

if __name__ == "__main__":
    main()