from flask import Flask, render_template, request, jsonify
import os
import re
from groq import Groq

app = Flask(__name__)

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
    
    return {
        'original': question,
        'lowercase': question_lower,
        'cleaned': question_clean,
        'tokens': tokens
    }

def query_llm(question, api_key):
    """
    Sends the question to Groq LLM API and returns the response
    """
    try:
        client = Groq(api_key=api_key)
        
        # Construct prompt
        prompt = f"Answer the following question clearly and concisely: {question}"
        
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

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question submission"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a valid question'}), 400
        
        # Get API key from environment variable
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            return jsonify({'error': 'API key not configured'}), 500
        
        # Preprocess the question
        preprocessing = preprocess_question(question)
        
        # Query the LLM
        answer = query_llm(question, api_key)
        
        return jsonify({
            'success': True,
            'preprocessing': preprocessing,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use environment port or default to 5000
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)