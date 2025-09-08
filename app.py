from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from flask_cors import CORS
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
CORS(app)
load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalchatbot"

# Load Existing index 

docsearch= PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, max_output_tokens=500)

system_prompt = (
   "You are a knowledgeable assistant trained to answer questions using retrieved information. "
    "Base your answer strictly on the provided context. "
    "If the information is not available, respond with 'I don't know.' "
    "Keep your answers clear and concise (maximum three sentences)."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json
    user_input = msg.get("message", "")

    response = rag_chain.invoke({"input": user_input})
    answer = response.get("answer") or response.get("output_text", "Sorry, I couldn’t generate a response.")

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)