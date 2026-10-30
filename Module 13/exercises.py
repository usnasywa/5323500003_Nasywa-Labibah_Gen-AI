import os
from typing import TypedDict, Annotated, List, Literal
import operator
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("XKIRO_KEY")
if not api_key:
    raise ValueError("API Key (OPENROUTER_API_KEY / XKIRO_KEY) tidak ditemukan di .env!")

# Inisialisasi Chat Model (menggunakan gpt-4o-mini via OpenRouter API)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=512,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# EXERCISE 1: Simple Sequential Graph (State & Basic Flow)
# 1. Definisi State sederhananya
class SimpleState(TypedDict):
    input_text: str
    uppercase_text: str
    word_count: int

# 2. Node Functions
def uppercase_node(state: SimpleState) -> dict:
    text = state["input_text"]
    return {"uppercase_text": text.upper()}

def count_words_node(state: SimpleState) -> dict:
    text = state["uppercase_text"]
    count = len(text.split())
    return {"word_count": count}

# 3. Build Graph
builder = StateGraph(SimpleState)
builder.add_node("uppercase", uppercase_node)
builder.add_node("count_words", count_words_node)

builder.add_edge(START, "uppercase")
builder.add_edge("uppercase", "count_words")
builder.add_edge("count_words", END)

ex1_graph = builder.compile()

# 4. Invoke Graph
ex1_input = {"input_text": "langgraph mempermudah pembuatan alur kerja agentik"}
ex1_result = ex1_graph.invoke(ex1_input)

print(f"Input Text      : {ex1_result['input_text']}")
print(f"Uppercase Text  : {ex1_result['uppercase_text']}")
print(f"Word Count      : {ex1_result['word_count']}")

# EXERCISE 2: Conditional Routing (Branching Logic)
class RouteState(TypedDict):
    query: str
    intent: str
    response: str

# Node Classifier (Menentukan Intent)
def classify_intent_node(state: RouteState) -> dict:
    q = state["query"].lower()
    if any(w in q for w in ["kode", "code", "python", "fungsi"]):
        intent = "code"
    elif any(w in q for w in ["salam", "halo", "hi"]):
        intent = "greeting"
    else:
        intent = "general"
    return {"intent": intent}

# Specialized Handler Nodes
def code_handler(state: RouteState) -> dict:
    msg = [
        SystemMessage(content="Anda adalah pakar pemrograman Python. Berikan penjelasan singkat."),
        HumanMessage(content=state["query"])
    ]
    res = llm.invoke(msg)
    return {"response": f"[Code Branch] {res.content}"}

def greeting_handler(state: RouteState) -> dict:
    return {"response": "[Greeting Branch] Halo! Ada yang bisa saya bantu terkait pemrograman hari ini?"}

def general_handler(state: RouteState) -> dict:
    res = llm.invoke([HumanMessage(content=state["query"])])
    return {"response": f"[General Branch] {res.content}"}

# Conditional Router Function
def route_intent(state: RouteState) -> Literal["code_node", "greeting_node", "general_node"]:
    intent = state["intent"]
    if intent == "code":
        return "code_node"
    elif intent == "greeting":
        return "greeting_node"
    return "general_node"

# Build Graph
route_builder = StateGraph(RouteState)
route_builder.add_node("classifier", classify_intent_node)
route_builder.add_node("code_node", code_handler)
route_builder.add_node("greeting_node", greeting_handler)
route_builder.add_node("general_node", general_handler)

route_builder.add_edge(START, "classifier")
route_builder.add_conditional_edges("classifier", route_intent)
route_builder.add_edge("code_node", END)
route_builder.add_edge("greeting_node", END)
route_builder.add_edge("general_node", END)

ex2_graph = route_builder.compile()

# Test Routing
res_code = ex2_graph.invoke({"query": "Tuliskan fungsi Python untuk membalik string."})
print(f"Query  : Tuliskan fungsi Python...")
print(f"Intent : {res_code['intent']}")
print(f"Output : {res_code['response']}\n")

res_greet = ex2_graph.invoke({"query": "Halo, selamat pagi!"})
print(f"Query  : Halo, selamat pagi!")
print(f"Intent : {res_greet['intent']}")
print(f"Output : {res_greet['response']}")

# EXERCISE 3: Iterative Loop (Reflection & Revision Loop)
class RevisionState(TypedDict):
    draft: str
    critique: str
    revision_count: int

def generate_draft_node(state: RevisionState) -> dict:
    prompt = "Tuliskan ringkasan 1 paragraf tentang pentingnya RAG dalam AI."
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": res.content, "revision_count": 1}

def critique_node(state: RevisionState) -> dict:
    prompt = f"Evaluasi draf berikut secara singkat. Jika draf sudah cukup baik dan memiliki lebih dari 20 kata, jawab 'BAGUS'. Jika belum, berikan 1 masukan singkat:\n\nDraf:\n{state['draft']}"
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": res.content}

def revise_node(state: RevisionState) -> dict:
    prompt = f"Perbaiki draf berikut berdasarkan masukan:\nMasukan: {state['critique']}\n\nDraf Lama:\n{state['draft']}"
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": res.content, "revision_count": state["revision_count"] + 1}

def should_continue(state: RevisionState) -> Literal["revise_node", "__end__"]:
    # Loop maksimal 2 kali revisi atau berhenti jika kritik sudah 'BAGUS'
    if "BAGUS" in state["critique"].upper() or state["revision_count"] >= 2:
        return END
    return "revise_node"

# Build Graph Loop
loop_builder = StateGraph(RevisionState)
loop_builder.add_node("generate_draft", generate_draft_node)
loop_builder.add_node("critique_node", critique_node)
loop_builder.add_node("revise_node", revise_node)

loop_builder.add_edge(START, "generate_draft")
loop_builder.add_edge("generate_draft", "critique_node")
loop_builder.add_conditional_edges("critique_node", should_continue)
loop_builder.add_edge("revise_node", "critique_node")

ex3_graph = loop_builder.compile()

res_loop = ex3_graph.invoke({})
print(f"Jumlah Revisi : {res_loop['revision_count']}")
print(f"Kritik Terakhir: {res_loop['critique']}")
print(f"Draf Akhir    : {res_loop['draft']}")

# EXERCISE 4: Agent State with Messages Reducer (Chat Memory State)
# Menggunakan Annotated[list, operator.add] agar pesan-pesan baru otomatis ditambahkan (append)
class AgentMessageState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def bot_node(state: AgentMessageState) -> dict:
    # Memanggil model dengan seluruh histori pesan yang tersimpan di State
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build Conversation Graph
chat_builder = StateGraph(AgentMessageState)
chat_builder.add_node("bot", bot_node)
chat_builder.add_edge(START, "bot")
chat_builder.add_edge("bot", END)

ex4_graph = chat_builder.compile()

# Putaran 1 Percakapan
state_input_1 = {"messages": [HumanMessage(content="Nama saya Labibah.")]}
state_output_1 = ex4_graph.invoke(state_input_1)

print(f"User : Nama saya Labibah.")
print(f"Bot  : {state_output_1['messages'][-1].content}\n")

# Putaran 2 Percakapan (Meneruskan histori dari state_output_1)
state_input_2 = {
    "messages": state_output_1["messages"] + [HumanMessage(content="Siapa nama saya?")]
}
state_output_2 = ex4_graph.invoke(state_input_2)

print(f"User : Siapa nama saya?")
print(f"Bot  : {state_output_2['messages'][-1].content}")