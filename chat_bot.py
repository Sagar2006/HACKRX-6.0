import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

print("🤖 AI Assistant - Ask me anything! (Type 'quit' to exit)")
print("-" * 50)

# Store conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
]

while True:
    # Get user input
    user_question = input("\n💭 You: ").strip()
    
    # Check if user wants to quit
    if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not user_question:
        print("⚠️ Please enter a question!")
        continue
    
    # Add user message to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_question,
    })
    
    try:
        # Get AI response
        response = client.chat.completions.create(
            messages=conversation_history,
            temperature=1.0,
            top_p=1.0,
            model=model
        )
        
        ai_response = response.choices[0].message.content
        
        # Add AI response to conversation history
        conversation_history.append({
            "role": "assistant", 
            "content": ai_response
        })
        
        print(f"\n🤖 AI: {ai_response}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Remove the last user message if there was an error
        conversation_history.pop()
