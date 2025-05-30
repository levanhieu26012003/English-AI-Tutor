import os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

class EnglishTutorAgent:
    def __init__(self):
        """Initialize the English Tutor Agent with Groq"""
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.conversation_history = []
        self.user_profile = {
            'level': 'beginner',  # beginner, intermediate, advanced
            'focus_areas': ['grammar', 'vocabulary', 'conversation'],
            'native_language': 'vietnamese',
            'goals': 'general_english'
        }
        # Các model miễn phí tốt trên Groq
        self.available_models = {
            'fast': 'llama-3.1-8b-instant',      # Nhanh nhất, tốt cho hội thoại
            'balanced': 'llama3-70b-8192', 
            'smart': 'mixtral-8x7b-32768',       # Thông minh, context dài
            'coding': 'llama-3.2-90b-text-preview' # Tốt nhất cho giải thích
        }
        self.current_model = self.available_models['balanced']  # Model mặc định
        
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
- Đặt câu hỏi để duy trì cuộc trò chuyện

Hãy bắt đầu bằng cách chào hỏi và hỏi về mục tiêu học tập của học viên."""

    def chat(self, user_message):
        """Process user message and return AI response"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Prepare messages for API call
            messages = [
                {"role": "system", "content": self.set_system_prompt()}
            ]
            
            # Add recent conversation history (last 10 messages to manage token limit)
            recent_history = self.conversation_history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                temperature=0.7,  # Creativity level
                max_tokens=500,   # Limit response length
                stream=False      # Không stream để đơn giản
            )
            
            # Extract AI response
            ai_response = response.choices[0].message.content
            
            # Add AI response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return ai_response
            
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra: {str(e)}"
    
    def analyze_text(self, text):
        """Analyze user's English text for errors and improvements"""
        analysis_prompt = f"""Phân tích đoạn text tiếng Anh này của học viên:

Text: "{text}"

Hãy cung cấp:
1. Lỗi ngữ pháp (nếu có) và cách sửa
2. Gợi ý từ vựng tốt hơn
3. Cải thiện cấu trúc câu
4. Điểm đánh giá tổng thể (1-10)
5. Lời khuyến khích

Trả lời bằng tiếng Việt để học viên dễ hiểu."""

        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích tiếng Anh."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Không thể phân tích text: {str(e)}"
    
    def switch_model(self, model_tier='balanced'):
        """Switch between different model tiers"""
        if model_tier in self.available_models:
            self.current_model = self.available_models[model_tier]
            return f"Đã chuyển sang model: {model_tier} ({self.current_model})"
        return "Model tier không hợp lệ. Chọn: fast, balanced, smart, coding"
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            'current_model': self.current_model,
            'available_models': self.available_models,
            'model_info': {
                'fast': 'llama-3.1-8b-instant - Nhanh nhất, phù hợp hội thoại',
                'balanced': 'llama-3.1-70b-versatile - Cân bằng tốc độ và chất lượng',
                'smart': 'mixtral-8x7b-32768 - Thông minh, context dài',
                'coding': 'llama-3.2-90b-text-preview - Tốt nhất cho giải thích phức tạp'
            },
            'cost': '🎉 TẤT CẢ ĐỀU MIỄN PHÍ!'
        }
    
    def update_user_level(self, new_level):
        """Update user's English level"""
        if new_level in ['beginner', 'intermediate', 'advanced']:
            self.user_profile['level'] = new_level
            return f"Đã cập nhật trình độ thành: {new_level}"
        return "Trình độ không hợp lệ"
    
print("✅ File tutor_agent.py đã được import")
