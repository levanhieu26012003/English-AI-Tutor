import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Import LangChain components (cập nhật)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory # Dùng ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser # Output parser
# Không cần RunnablePassthrough, RunnableLambda nữa cho cách này

# Load environment variables
load_dotenv()

class EnglishTutorAgent:
    def __init__(self):
        """Initialize the English Tutor Agent with Google Gemini and LangChain Memory using LCEL."""
        
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
        
        genai.configure(api_key=gemini_api_key)
        
        self.user_profile = {
            'level': 'beginner',
            'focus_areas': ['grammar', 'vocabulary', 'conversation'],
            'native_language': 'vietnamese',
            'goals': 'general_english'
        }
        
        self.available_models = {
            'fast': 'gemini-2.0-flash',
            'balanced': 'gemini-2.0-flash',
            'smart': 'gemini-2.0-flash',
            'coding': 'gemini-2.0-flash'
        }
        self.current_model_name = self.available_models['balanced']
        
        # Khởi tạo LLM cho LangChain
        self.llm = ChatGoogleGenerativeAI(model=self.current_model_name, temperature=0.7)
        
        # Khởi tạo Memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            input_key="human_input",
            return_messages=True,
            k=5
        )

        # Khởi tạo Chain ban đầu với LCEL (sẽ không dùng .with_history() nữa)
        self._initialize_chain_lcel()
        
    def _initialize_chain_lcel(self):
        """Initializes the LangChain Chain with current LLM and memory using LCEL."""
        system_prompt_content = self.set_system_prompt()

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{human_input}")
        ])
        
        # Xây dựng Chain với LCEL. KHÔNG DÙNG .with_history() ở đây nữa.
        # Logic tải/lưu history sẽ được thực hiện khi gọi invoke().
        self.conversation_chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def set_system_prompt(self):
        """Create system prompt based on user profile"""
        level_descriptions = {
            'beginner': 'sử dụng từ vựng đơn giản, ngữ pháp cơ bản',
            'intermediate': 'sử dụng cấu trúc câu phức tạp hơn, từ vựng đa dạng',
            'advanced': 'thảo luận các chủ đề phức tạp, sử dụng ngôn ngữ tinh tế'
        }
        
        return f"""Bạn là một giáo viên tiếng Anh AI thân thiện và kiên nhẫn. 

Học viên của bạn:
- Trình độ: {self.user_profile['level']} ({level_descriptions[self.user_profile['level']]})
- Ngôn ngữ mẹ đẻ: Tiếng Việt
- Mục tiêu: {self.user_profile['goals']}

Nhiệm vụ của bạn:
1. Trò chuyện tự nhiên bằng tiếng Anh với học viên
2. Điều chỉnh độ khó phù hợp với trình độ
3. Sửa lỗi một cách nhẹ nhàng và giải thích rõ ràng
4. Khuyến khích và động viên học viên
5. Đưa ra gợi ý cải thiện cụ thể
6. Nếu học viên nói tiếng Việt, hãy khuyến khích họ thử nói tiếng Anh

Phong cách:
- Thân thiện, kiên nhẫn
- Sử dụng ví dụ thực tế
- Giải thích bằng cả tiếng Anh và tiếng Việt khi cần
- Đặt câu hỏi để duy trì cuộc trò chuyện"""

    def chat_with_langchain(self, user_message):
        """Process user message using the LangChain Chain and return AI response."""
        try:
            # 1. Tải lịch sử cuộc trò chuyện từ memory
            history = self.memory.load_memory_variables({})["chat_history"]

            # 2. Chuẩn bị input cho chain
            inputs = {"human_input": user_message, "chat_history": history}

            # 3. Gọi Chain với input và lịch sử
            response = self.conversation_chain.invoke(inputs)
            
            # 4. Lưu tin nhắn hiện tại và phản hồi vào memory SAU KHI nhận được response
            self.memory.save_context(
                {"human_input": user_message},
                {"output": response}
            )
            
            return response
            
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra khi trò chuyện: {str(e)}. Vui lòng thử lại hoặc kiểm tra cấu hình."
    
    def analyze_text(self, text):
        """Analyze user's English text for errors and improvements using a direct LLM call."""
        analysis_prompt = f"""Bạn là chuyên gia phân tích tiếng Anh. Phân tích đoạn text tiếng Anh này của học viên:

Text: "{text}"

Hãy cung cấp:
1. Lỗi ngữ pháp (nếu có) và cách sửa
2. Gợi ý từ vựng tốt hơn
3. Cải thiện cấu trúc câu
4. Điểm đánh giá tổng thể (1-10)
5. Lời khuyến khích

Trả lời bằng tiếng Việt để học viên dễ hiểu."""

        try:
            model_instance = genai.GenerativeModel(self.available_models['balanced']) # Dùng model cân bằng cho phân tích
            response = model_instance.generate_content(
                [
                    {"role": "user", "parts": [{"text": analysis_prompt}]}
                ],
                generation_config=genai.GenerationConfig(
                    temperature=0.3, # Ít sáng tạo, tập trung vào độ chính xác
                    max_output_tokens=800
                )
            )
            
            return response.text
            
        except Exception as e:
            return f"Không thể phân tích text: {str(e)}. Vui lòng thử lại."
    
    def switch_model(self, model_tier='balanced'):
        """Switch between different model tiers and re-initialize the Chain."""
        if model_tier in self.available_models:
            self.current_model_name = self.available_models[model_tier]
            self.llm = ChatGoogleGenerativeAI(model=self.current_model_name, temperature=0.7)
            
            # Cần khởi tạo lại Chain khi đổi model
            self._initialize_chain_lcel() 
            
            return f"Đã chuyển sang model: {model_tier} ({self.current_model_name})"
        return "Model tier không hợp lệ. Chọn: fast, balanced, smart, coding"
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            'current_model': self.current_model_name,
            'available_models': self.available_models,
            'model_info': {
                'fast': 'gemini-2.0-flash - Nhanh nhất, phù hợp hội thoại',
                'balanced': 'gemini-2.0-flash - Cân bằng tốc độ và chất lượng',
                'smart': 'gemini-2.0-flash - Thông minh, context dài',
                'coding': 'gemini-2.0-flash - Tốt nhất cho giải thích phức tạp'
            },
            'cost': 'Phụ thuộc vào gói sử dụng Gemini API của bạn (có thể có tier miễn phí giới hạn).'
        }
    
    def update_user_level(self, new_level):
        """Update user's English level and re-initialize the Chain (as system prompt depends on level)."""
        if new_level in ['beginner', 'intermediate', 'advanced']:
            self.user_profile['level'] = new_level
            # Cần khởi tạo lại Chain để cập nhật system prompt dựa trên trình độ mới
            self._initialize_chain_lcel()
            return f"Đã cập nhật trình độ thành: {new_level}"
        return "Trình độ không hợp lệ"