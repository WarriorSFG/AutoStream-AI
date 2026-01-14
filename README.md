# AutoStream-AI
# Social-to-Lead Agentic Workflow (AutoStream)

This repository contains an intelligent Conversational AI Agent built for **AutoStream**, a fictional video editing SaaS. The agent is designed to qualify leads from social media conversations using **LangGraph** for state management and **Gemini 2.5 Flash** for reasoning.

## 📌 Features
* [cite_start]**Intent Recognition:** Distinguishes between casual greetings, product inquiries, and high-intent buying signals. [cite: 20]
* [cite_start]**RAG (Retrieval-Augmented Generation):** accurate answers about pricing and policies using a local knowledge base (`data.json`). [cite: 25]
* [cite_start]**Contextual Memory:** Remembers user details (Name, Email) across multiple turns using persistent state. [cite: 89]
* [cite_start]**Safe Tool Execution:** only triggers the lead capture tool once *all* required slots (Name, Email, Platform) are filled. [cite: 44, 54]

## 🛠️ Tech Stack
* [cite_start]**Framework:** LangChain & LangGraph [cite: 79]
* [cite_start]**LLM:** Google Gemini 2.5 Flash [cite: 85]
* **Embeddings:** Google Gemini Embedding 1.0
* **Vector Store:** Scikit-Learn VectorStore (Local)
* [cite_start]**Language:** Python 3.13 [cite: 76]

---

## 🚀 How to Run Locally

### 1. Clone the Repository
git clone [https://github.com/YourUsername/AutoStream-AI.git](https://github.com/YourUsername/AutoStream-AI.git)

cd AutoStream-AI
2. Set Up Virtual Environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

5. Configure Environment Variables
Create a .env file in the root directory and add your Google API key:
GOOGLE_API_KEY=your_actual_api_key_here

5. Run the Agent
This script runs a simulated conversation showing RAG, Context Retention, and Tool Calling.
python agent.py

🧠 Architecture Explanation
Why LangGraph? For this project, I chose LangGraph over standard chains because the requirement demanded complex state management.

Cyclic State: Unlike a DAG (Directed Acyclic Graph), this agent needs to "loop" back to the user to ask clarifying questions (e.g., asking for a name, then an email) while retaining the context of previous answers.


Persistence: The MemorySaver checkpointer ensures that variables like name and email are stored in the state dict across conversation turns, preventing the bot from "forgetting" data. 

Conditional Logic: We use conditional edges to determine if the agent should respond to the user OR call a tool. The ToolNode is strictly gated; it only executes when the LLM determines all 3 lead parameters are present.

📱 WhatsApp Integration Strategy
To deploy this agent on WhatsApp, I would implement a Webhook Architecture using the Meta Cloud API: 
Server: I would wrap the agent.py logic in a FastAPI or Flask service exposing a POST /webhook endpoint.
Verification: Configure the endpoint in the Meta Developer Portal and handle the specific hub.challenge verification request.
Message Handling:
Incoming WhatsApp messages are received as JSON payloads.
The user's phone number acts as the unique thread_id for the LangGraph memory.
The text payload is passed to app.invoke().
Response: The agent's text response is sent back via a POST request to the Graph API: https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages
