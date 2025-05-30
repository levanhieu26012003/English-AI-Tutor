from agent.tutor_agent import EnglishTutorAgent

def main():
    """Simple command line interface for testing"""
    print("=== English AI Tutor ===")
    print("Gõ 'quit' để thoát")
    print("Gõ 'analyze: [text]' để phân tích văn bản")
    print("Gõ 'level: [beginner/intermediate/advanced]' để thay đổi trình độ")
    print("Gõ 'model: [fast/balanced/smart/coding]' để thay đổi model")
    print("Gõ 'info' để xem thông tin model")
    print("-" * 50)
    
    # Initialize agent
    tutor = EnglishTutorAgent()
    
    # Start conversation
    print("AI:", tutor.chat("Hello! Let's start our English conversation."))
    
    while True:
        user_input = input("\nBạn: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nTạm biệt! Chúc bạn học tốt!")
            break
            
        elif user_input.startswith('analyze:'):
            text_to_analyze = user_input[8:].strip()
            analysis = tutor.analyze_text(text_to_analyze)
            print(f"\nPhân tích:\n{analysis}")
            
        elif user_input.startswith('level:'):
            new_level = user_input[6:].strip()
            result = tutor.update_user_level(new_level)
            print(f"\n{result}")
            
        elif user_input.startswith('model:'):
            model_tier = user_input[6:].strip()
            result = tutor.switch_model(model_tier)
            print(f"\n{result}")
            
        elif user_input == 'info':
            info = tutor.get_model_info()
            print(f"\nModel hiện tại: {info['current_model']}")
            print(f"Chi phí: {info['cost']}")
            print("Các model có sẵn:")
            for tier, desc in info['model_info'].items():
                print(f"  - {tier}: {desc}")
            
        else:
            response = tutor.chat(user_input)
            print(f"\nAI: {response}")

if __name__ == "__main__":
    main()