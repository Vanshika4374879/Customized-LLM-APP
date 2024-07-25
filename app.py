import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("TECH.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are Troubleshooter 🖥️. You'll do your best to help me resolve my issue. Whether it's troubleshooting software problems, hardware issues, or general tech queries, You are here to assist you!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response



demo = gr.Blocks()

with demo:
    gr.Markdown(
        "‼️Disclaimer: This document is intended solely for the implementation of a Retrieval-Augmented Generation (RAG) chatbot. ‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
       examples=[
            ["My computer is running slow. How can I speed it up? 🖥️"],
            ["I'm getting an error message when I try to install software. What should I do? 💻"],
            ["How do I connect my printer wirelessly to my computer? 🖨️"],

        ],
        
        title='Tech Troubleshooter 🖥️',
    description='''<div style="text-align: left; font-family: Arial, sans-serif; color: #333;">
                   <h2>Welcome to the Tech Troubleshooter 🖥️</h2>
                   <p style="font-size: 16px; text-align: left;">Please describe the technical issue you're facing, and I'll do my best to provide assistance.</p>
                   <p style="text-align: left;"><strong>Examples:</strong></p>
                   <ul style="list-style-type: disc; margin-left: 20px; text-align: left;">
                       <li style="font-size: 14px;">My computer is running slow. How can I speed it up? 🖥️</li>
                       <li style="font-size: 14px;">I'm getting an error message when I try to install software. What should I do? 💻</li>
                       <li style="font-size: 14px;">How do I connect my printer wirelessly to my computer? 🖨️</li>
                   </ul>
                   </div>''',
    )


if __name__ == "__main__":
    demo.launch()
