import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory 
from langchain_core.output_parsers import StrOutputParser 
from langchain.agents import AgentExecutor, create_react_agent, Tool
from wikipedia import exceptions as wikipedia_exceptions # Thêm import này
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Google Calendar Tools
from urllib.parse import quote_plus
import pytz
import json

# Load environment variables
load_dotenv()

class RAGTool:
    def __init__(self):
        DOC_PATH = os.path.join(os.path.dirname(__file__), '..', 'docs', 'doc.txt')

        with open(DOC_PATH, "r", encoding="utf-8") as f:
            document_text = f.read()
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
        )   
        chunks = splitter.split_text(document_text)
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(chunks, embedding_model)
        self.vectorstore = vectorstore

    def run(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])


class WikipediaTool:
    def __init__(self):
        self.wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

    def run(self, query: str) -> str:
        """Tìm kiếm thông tin trên Wikipedia. Sử dụng khi cần tra cứu các khái niệm, từ vựng, sự kiện. 
        Đầu vào là một chuỗi truy vấn."""
        try:
            return self.wrapper.run(query)
        except wikipedia_exceptions.PageError:
            return "Không tìm thấy thông tin trên Wikipedia cho truy vấn này."
        except wikipedia_exceptions.DisambiguationError as e:
            return f"Truy vấn quá mơ hồ. Vui lòng cụ thể hơn. Các lựa chọn: {e.options[:5]}..."
        except Exception as e:
            return f"Có lỗi khi tìm kiếm Wikipedia: {str(e)}"
    
class CurrentDateTimeTool:
    def run(self, _: str) -> str: 
        """Trả về ngày và giờ hiện tại."""
        return datetime.now().strftime("Ngày: %A, %d/%m/%Y - Giờ: %H:%M:%S")        
     
class GoogleCalendarAddEventTool:
    def run(self, tool_input: str) -> str: # Hàm run chỉ nhận DUY NHẤT một chuỗi tool_input
        """
        Tạo một liên kết để thêm sự kiện vào Google Calendar.
        Sử dụng khi người dùng muốn đặt lịch học hoặc một sự kiện cụ thể.
        Đầu vào tool_input (str) phải là một chuỗi JSON chứa các khóa:
        - "title" (str): Tiêu đề sự kiện (bắt buộc).
        - "start_datetime_str" (str): Thời gian bắt đầu sự kiện (bắt buộc, ví dụ: "2025-12-25 10:00").
        - "duration_minutes" (int, optional): Thời lượng sự kiện bằng phút (mặc định 60 phút).
        - "description" (str, optional): Mô tả chi tiết sự kiện.
        """
        try:
            # Phân tích cú pháp chuỗi JSON đầu vào
            params = json.loads(tool_input)
            
            # Trích xuất các tham số từ dictionary
            title = params.get("title")
            start_datetime_str = params.get("start_datetime_str")
            duration_minutes = params.get("duration_minutes", 60) # Mặc định 60 nếu không có
            description = params.get("description", "") # Mặc định rỗng nếu không có

            # Kiểm tra các tham số bắt buộc
            if not title or not start_datetime_str:
                return "Lỗi: Đầu vào JSON thiếu 'title' hoặc 'start_datetime_str' bắt buộc."

            # --- Phần còn lại của logic xử lý ngày giờ và tạo URL (giữ nguyên) ---
            parsed_start_dt = None
            formats_to_try = [
                "%Y-%m-%d %H:%M",  # "2025-12-25 10:00"
                "%d/%m/%Y %H:%M",  # "25/12/2025 10:00"
                "%d-%m-%Y %H:%M",  # "25-12-2025 10:00"
                "%Y-%m-%d %H:%M:%S", # "2025-12-25 10:00:00"
            ]
            
            for fmt in formats_to_try:
                try:
                    parsed_start_dt = datetime.strptime(start_datetime_str, fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_start_dt is None:
                return f"Không thể hiểu định dạng thời gian bắt đầu: '{start_datetime_str}'. Vui lòng cung cấp định dạng rõ ràng hơn như 'YYYY-MM-DD HH:MM'."

            local_tz = pytz.timezone('Asia/Ho_Chi_Minh') 
            local_dt = local_tz.localize(parsed_start_dt, is_dst=None)
            utc_dt = local_dt.astimezone(pytz.utc)

            end_dt = utc_dt + timedelta(minutes=duration_minutes)

            start_time_gcal = utc_dt.strftime("%Y%m%dT%H%M%SZ")
            end_time_gcal = end_dt.strftime("%Y%m%dT%H%M%SZ")

            base_url = "https://calendar.google.com/calendar/render"
            gcal_params = { # Đổi tên biến để tránh trùng với 'params' đã phân tích
                "action": "TEMPLATE",
                "text": title,
                "dates": f"{start_time_gcal}/{end_time_gcal}",
                "details": description,
                "sf": "true",
                "output": "xml" 
            }

            query_string = "&".join([f"{key}={quote_plus(str(value))}" for key, value in gcal_params.items()])
            
            return f"Bạn có thể thêm sự kiện '{title}' vào lịch của mình bằng liên kết này: {base_url}?{query_string}"
        except json.JSONDecodeError:
            return "Lỗi: Đầu vào cho công cụ Lịch không phải là định dạng JSON hợp lệ. Vui lòng kiểm tra lại cấu trúc JSON."
        except Exception as e:
            return f"Có lỗi khi tạo liên kết lịch: {str(e)}. Vui lòng kiểm tra định dạng thời gian hoặc các tham số."
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
        self.llm = ChatGoogleGenerativeAI(model=self.current_model_name, temperature=0.3)
        
        # Khởi tạo Memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            input="human_input",
            return_messages=True,
            k=5
        )
        
        self.tools = [
            Tool(
                name="Wikipedia Search",
                func=WikipediaTool().run,
                description="Hữu ích khi bạn cần tra cứu thông tin chung, khái niệm, hoặc sự kiện. Sử dụng nó để tìm kiếm các chủ đề mà người dùng hỏi đến.",
            ),
            Tool(
                name="Current Time",
                func=CurrentDateTimeTool().run,
                description="Hữu ích khi bạn cần biết ngày và giờ hiện tại.",
            ),
            Tool(
                name="Google Calendar Add Event",
                func=GoogleCalendarAddEventTool().run,
                description="Hữu ích khi người dùng muốn đặt lịch học hoặc một sự kiện. Cần các tham số: title (tiêu đề sự kiện), start_datetime_str (thời gian bắt đầu, ví dụ: '2025-12-25 10:00'), duration_minutes (thời lượng bằng phút, mặc định 60), description (mô tả).",
            ),
            Tool(
            name="EnglishMaterialSearch",
            func=RAGTool().run,
            description="Trả lời các câu hỏi dựa trên tài liệu học tiếng Anh"
        )
        ]

        # Khởi tạo Chain ban đầu với LCEL (sẽ không dùng .with_history() nữa)
        self._initialize_agent()
        
    def _initialize_agent(self):
        """
        Initializes the LangChain Agent with the current LLM, tools, and memory.
        This method sets up the Agent's prompt, creates the ReAct agent,
        and initializes the AgentExecutor which orchestrates the agent's reasoning,
        tool usage, and memory management.
        """

        # Custom system prompt for the Agent, based on user profile
        system_prompt = self.set_system_prompt()

        # Define the prompt template for the Agent.
        # This template guides the LLM on its role, available tools,
        # how to use them (JSON format), and includes chat history and current input.
        # IMPORTANT: "tools", "chat_history", "input", and "agent_scratchpad"
        # are required placeholders for a ReAct agent.
        prompt = ChatPromptTemplate.from_template(
            f"""({system_prompt})Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

            {{tools}}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do.
            (If you need to use a tool, follow the Action/Observation format. Otherwise, proceed directly to Final Answer.)
            Action: the action to take, should be one of [{{tool_names}}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin! Remember to speak as a compasionate medical professional when giving your final answer. If the condition is serious advise they speak to a doctor.

            Previous conversation history:
            {{chat_history}}

            User question: {{input}}
            {{agent_scratchpad}}""")
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )
    
    def run_agent_chat(self, user_message):
        """Process user message using the LangChain Agent and return AI response."""
        try:
            response = self.agent_executor.invoke({"input": user_message})
            return response['output']

        except Exception as e:
            print(f"\n--- LỖI TRONG run_agent_chat: ---\n{e}\n-----------------------------------\n")
            return f"Xin lỗi, có lỗi xảy ra khi trò chuyện: {str(e)}. Vui lòng thử lại hoặc kiểm tra cấu hình."
        
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
7. Nếu cần, hãy sử dụng các công cụ được cung cấp để tra cứu thông tin hoặc lấy dữ liệu, bao gồm cả việc tạo liên kết để đặt lịch học.

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
            self.llm = ChatGoogleGenerativeAI(model=self.current_model_name, temperature=0.3)
            
            # Cần khởi tạo lại Chain khi đổi model
            self._initialize_agent() 
            
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
            self._initialize_agent()
            return f"Đã cập nhật trình độ thành: {new_level}"
        return "Trình độ không hợp lệ"