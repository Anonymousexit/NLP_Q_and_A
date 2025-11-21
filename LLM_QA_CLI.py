import os
import re
from groq import Groq

def preprocess_question(question):
    """
    Preprocesses the input question by:
    - Converting to lowercase
    - Removing punctuation
    - Tokenizing
    """
    # Convert to lowercase
    question_lower = question.lower()
    
    # Remove punctuation (keep letters, numbers, and spaces)
    question_clean = re.sub(r'[^\w\s]', '', question_lower)
    
    # Tokenize (split into words)
    tokens = question_clean.split()
    
    print(f"\n[Preprocessing]")
    print(f"Original: {question}")
    print(f"Lowercase: {question_lower}")
    print(f"Cleaned: {question_clean}")
    print(f"Tokens: {tokens}")
    
    return question_clean, tokens

def query_llm(question, api_key):
    """
    Sends the question to Groq LLM API and returns the response
    """
    try:
        client = Groq(api_key=api_key)
        
        # Construct prompt
        prompt = f"Answer the following question clearly and concisely: {question}"
        
        print(f"\n[Sending to LLM...]")
        
        # Make API call
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        
        response = completion.choices[0].message.content
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """
    Main function for CLI application
    """
    print("="*60)
    print("    LLM-POWERED Q&A SYSTEM - CLI VERSION")
    print("="*60)
    
    # Get API key from environment or user input
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("\n[Setup Required]")
        api_key = input("Enter your Groq API Key: ").strip()
        
        if not api_key:
            print("Error: API key is required!")
            return
    
    print("\nType 'quit' or 'exit' to end the session.\n")
    
    while True:
        # Get user question
        question = input("\n🤔 Enter your question: ").strip()
        
        # Check for exit commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Q&A system. Goodbye! 👋")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        # Preprocess the question
        cleaned_question, tokens = preprocess_question(question)
        
        # Query the LLM
        answer = query_llm(question, api_key)
        
        # Display the answer
        print("\n" + "="*60)
        print("💡 ANSWER:")
        print("="*60)
        print(answer)
        print("="*60)

if __name__ == "__main__":
    main()