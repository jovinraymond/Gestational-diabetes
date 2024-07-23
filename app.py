from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import openai

# Initialize the Flask application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'supersecretkey'
Session(app)

# Load your faqs dataset
file_path = 'Gestational Diabetes Mellitus Screening.xlsx'
df = pd.read_excel(file_path)

# Prepare questions and answers
questions = df['Question'].fillna('').tolist()
Midwifery_Advice = df['Midwifery Advice / Medical solution'].fillna('').tolist()
Addressing_misinformation = df['Addressing misinformation'].fillna('').tolist()

# Load a pre-trained sentence transformer model
retriever = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = retriever.encode(questions, convert_to_tensor=True)

# Function to retrieve answer
def retrieve_answer(question, top_k=1):
    question_embedding = retriever.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, question_embeddings, top_k=top_k)

    if hits[0]:
        best_match_idx = hits[0][0]['corpus_id']
        return questions[best_match_idx], Midwifery_Advice[best_match_idx]
    else:
        return None, None

# OpenAI API key
openai.api_key = 'API KEY'

# Function to perform RAG with memory
def perform_rag(question):
    # Retrieve the conversation history from the session
    conversation_history = session.get('conversation_history', [])

    # Retrieving relevant context (answer) based on the question
    retrieved_question, context = retrieve_answer(question)

    if context:
        # Include previous context from conversation history
        history_context = " ".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Diabetes AI Advisor, an independent assistant providing information and guidance on diabetes."},
                {"role": "user", "content": f"Previous context: {history_context}"},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"Question: {question}"},
            ],
            max_tokens=200,
            temperature=0.7,
            n=1,
            stop=None
        )
        generated = response['choices'][0]['message']['content'].strip()

        # Update the conversation history
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": generated})
        session['conversation_history'] = conversation_history

        return generated
    else:
        return "Sorry, I don't have information on that question."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get('message')
    response_message = perform_rag(user_message)
    return jsonify({"response": response_message})

if __name__ == '__main__':
    app.run(debug=True)
