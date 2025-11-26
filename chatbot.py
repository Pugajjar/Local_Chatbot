import nltk
import string
import warnings
import random
import os
import pandas as pd
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from werkzeug.utils import secure_filename

# --- NEW IMPORTS ---
try:
    import PyPDF2
    from docx import Document
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
except ImportError as e:
    print(f"Warning: Missing library {e}. Some file features may not work.")

# --- CONFIGURATION & SETUP ---
app = Flask(__name__)
CORS(app)
warnings.filterwarnings('ignore')

# Download necessary NLTK data
resources = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'stopwords']
for resource in resources:
    try:
        if resource == 'punkt_tab': nltk.data.find('tokenizers/punkt_tab')
        elif resource == 'punkt': nltk.data.find('tokenizers/punkt')
        elif resource == 'stopwords': nltk.data.find('corpora/stopwords')
        else: nltk.data.find(f'corpora/{resource}')
    except LookupError:
        print(f"Downloading required NLTK resource: {resource}...")
        nltk.download(resource, quiet=True)

# --- CODE TEMPLATES LIBRARY (The "Code Engine") ---
CODE_TEMPLATES = {
    "addition": {
        "python": "def add(a, b):\n    return a + b\n\nnum1 = 5\nnum2 = 3\nprint(f'Sum: {add(num1, num2)}')",
        "cpp": "#include <iostream>\nusing namespace std;\n\nint main() {\n    int a = 5, b = 3;\n    cout << \"Sum: \" << a + b << endl;\n    return 0;\n}",
        "java": "public class Main {\n    public static void main(String[] args) {\n        int a = 5, b = 3;\n        System.out.println(\"Sum: \" + (a + b));\n    }\n}",
        "javascript": "function add(a, b) {\n    return a + b;\n}\n\nconsole.log('Sum:', add(5, 3));"
    },
    "factorial": {
        "python": "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)\n\nprint(factorial(5))",
        "cpp": "#include <iostream>\nusing namespace std;\n\nint factorial(int n) {\n    return (n == 0) ? 1 : n * factorial(n - 1);\n}\n\nint main() {\n    cout << factorial(5) << endl;\n    return 0;\n}",
        "java": "public class Main {\n    public static int factorial(int n) {\n        return (n == 0) ? 1 : n * factorial(n - 1);\n    }\n    public static void main(String[] args) {\n        System.out.println(factorial(5));\n    }\n}",
        "javascript": "function factorial(n) {\n    return n === 0 ? 1 : n * factorial(n - 1);\n}\nconsole.log(factorial(5));"
    },
    "palindrome": {
        "python": "def is_palindrome(s):\n    return s == s[::-1]\n\nprint(is_palindrome('madam'))",
        "cpp": "#include <iostream>\n#include <string>\n#include <algorithm>\nusing namespace std;\n\nbool isPalindrome(string s) {\n    string rev = s;\n    reverse(rev.begin(), rev.end());\n    return s == rev;\n}\n\nint main() {\n    cout << isPalindrome(\"madam\") << endl;\n    return 0;\n}",
        "java": "public class Main {\n    public static boolean isPalindrome(String s) {\n        return new StringBuilder(s).reverse().toString().equals(s);\n    }\n    public static void main(String[] args) {\n        System.out.println(isPalindrome(\"madam\"));\n    }\n}",
        "javascript": "function isPalindrome(s) {\n    return s === s.split('').reverse().join('');\n}\nconsole.log(isPalindrome('madam'));"
    }
}

# --- KNOWLEDGE BASE ---
knowledge_base = [
    {"tag": "greeting", "patterns": ["hi", "hello", "hey", "good morning"], "responses": ["Hello! Ready to work.", "Hi there! What are we building today?"]},
    {"tag": "capabilities", "patterns": ["what can you do", "help"], "responses": ["I can summarize PDFs, Excel sheets, and Word docs. I also have a Code Beautifier and can write simple programs (e.g., 'code for addition')!"]},
    {"tag": "creator", "patterns": ["who made you", "developer"], "responses": ["I am a custom local AI built with Python."]},
    {"tag": "bye", "patterns": ["bye", "exit", "quit"], "responses": ["Goodbye!", "Shutting down..."]}
]

# --- TEXT PROCESSING ---
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- SMART SUMMARIZER (TextRank) ---
def summarize_text(text, num_sentences=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        
        # Combine sentences
        summarized_text = " ".join([str(sentence) for sentence in summary])
        
        if not summarized_text:
            return "The document text was too short or unclear to summarize."
        return summarized_text
    except Exception as e:
        return f"Summarization failed: {str(e)}"

# --- FILE EXTRACTION HELPER (Now with Excel!) ---
def extract_text_from_file(file):
    filename = secure_filename(file.filename)
    text = ""
    
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
                
        elif filename.endswith('.docx'):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + " "
                
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            # Read Excel file into a pandas DataFrame
            df = pd.read_excel(file)
            # Convert to string (handling NaN values)
            text = df.to_string(index=False, na_rep=' ')
            
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
            text = df.to_string(index=False, na_rep=' ')
            
        else:
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
        
    return text.strip()

# --- TRAINING ---
corpus = []
tags = []
for intent in knowledge_base:
    for pattern in intent['patterns']:
        corpus.append(preprocess_text(pattern))
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def get_chat_response(user_input):
    user_input_processed = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vector, X)
    best_match_index = similarities.argmax()
    if similarities[0, best_match_index] < 0.2:
        return "I'm sorry, I don't understand that. Try asking about my capabilities."
    
    matched_tag = tags[best_match_index]
    for intent in knowledge_base:
        if intent['tag'] == matched_tag:
            return random.choice(intent['responses'])

def get_code_snippet(user_input):
    user_input = user_input.lower()
    
    # 1. Detect Topic
    topic = None
    for key in CODE_TEMPLATES.keys():
        if key in user_input:
            topic = key
            break
    
    if not topic:
        return None
        
    # 2. Detect Language
    language = "python" # Default
    if "c++" in user_input or "cpp" in user_input:
        language = "cpp"
    elif "java" in user_input and "script" not in user_input:
        language = "java"
    elif "javascript" in user_input or "js" in user_input:
        language = "javascript"
        
    # 3. Return formatted code
    code = CODE_TEMPLATES[topic].get(language, "Code not available for this language.")
    
    # Added {language} to the markdown fence so UI can detect it
    return f"**Here is the {language} code for {topic}:**\n\n```{language}\n{code}\n```"

# --- FLASK ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>AI Workstation Backend is Running!</h1>"

@app.route('/chat', methods=['POST'])
def chat():
    response_text = ""
    
    # 1. Handle File Upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                summary = summarize_text(extracted_text)
                response_text = f"**File Analysis:**\n\n{summary}"
            else:
                response_text = "I couldn't read that file. I support PDF, DOCX, TXT, CSV, and XLSX."

    # 2. Handle Text Message
    elif 'message' in request.form:
        user_message = request.form['message']
        
        # Check for code request first
        code_response = get_code_snippet(user_message)
        if code_response:
            response_text = code_response
        else:
            response_text = get_chat_response(user_message)

    else:
        return jsonify({"error": "No input provided"}), 400

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True, port=5000)