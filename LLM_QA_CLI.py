import os
import re
import google.generativeai as genai

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
    Sends the question to Gemini LLM API and returns the response
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Create model instance
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        # Construct prompt
        prompt = f"Answer the following question clearly and concisely: {question}"
        
        print(f"\n[Sending to LLM...]")
        
        # Make API call
        response = model.generate_content(prompt)
        
        return response.text
        
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
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\n[Setup Required]")
        api_key = input("Enter your Gemini API Key: ").strip()
        
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