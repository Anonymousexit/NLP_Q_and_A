from flask import Flask, render_template, request, jsonify
import os
import re
import google.generativeai as genai

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
    Sends the question to Gemini LLM API and returns the response
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Create model instance
        model = genai.GenerativeModel('gemini-pro')
        
        # Construct prompt
        prompt = f"Answer the following question clearly and concisely: {question}"
        
        # Make API call
        response = model.generate_content(prompt)
        
        return response.text
        
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
        api_key = os.getenv('GEMINI_API_KEY')
        
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