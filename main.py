from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# 1. Define tools
def suggest_assignment_group_fn(description: str) -> str:
    """Suggests the assignment group based on incident description."""
    desc = description.lower()
    if "payment" in desc or "gateway" in desc:
        return "Payment Team"
    elif "checkout" in desc or "cart" in desc:
        return "Frontend Web Team"
    elif "image" in desc or "product" in desc:
        return "Content Management"
    elif "login" in desc or "session" in desc:
        return "Authentication Services"
    elif "email" in desc or "order" in desc:
        return "Order Fulfillment"
    elif "coupon" in desc or "discount" in desc:
        return "Promotions Team"
    else:
        return "General IT Support"

suggest_assignment_group = Tool(
    name="suggest_assignment_group",
    func=suggest_assignment_group_fn,
    description="based on keyword from incident description, suggests the assignment group."
)

def suggest_action_fn(description: str) -> str:
    """Parses the description to extract priority and impact and return a triage action."""
    # crude parse for agent-friendly input format
    priority = "low"
    impact = "low"
    if "priority:" in description.lower() and "impact:" in description.lower():
        try:
            parts = description.lower().split("priority:")[1].strip().split("impact:")
            priority = parts[0].strip().strip('.')
            impact = parts[1].strip().strip('.')
        except Exception:
            pass

    if priority == "critical" or impact == "high":
        return "Escalate to Incident Manager"
    elif priority == "high" or impact == "medium":
        return "Immediate attention by team lead"
    else:
        return "Assign to appropriate group and monitor"

suggest_action = Tool(
    name="suggest_action",
    func=suggest_action_fn,
    description="Suggests next triage action based on priority and impact."
)

# 2. Load historical data and prepare embeddings
historical_df = pd.read_excel("data_source/historical_incidents.xlsx")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
historical_df["embedding"] = historical_df["Short Description"].apply(
    lambda x: embedder.encode(str(x), convert_to_numpy=True)
)

def historical_assignment_lookup_fn(description: str) -> str:
    """Finds assignment group from historical data using embedding similarity."""
    query_vec = embedder.encode(description, convert_to_numpy=True)
    similarities = cosine_similarity([query_vec], list(historical_df["embedding"]))
    best_match_index = int(np.argmax(similarities))
    return historical_df.iloc[best_match_index]["Assignment Group"]

historical_assignment_lookup = Tool(
    name="historical_assignment_lookup",
    func=historical_assignment_lookup_fn,
    description="Uses embedding similarity with past incident descriptions to suggest the best assignment group."
)


# 3. Define LLM
llm = ChatOpenAI(model="gpt-4o")

# 4. Define custom prompt with agent_scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a triage assistant for IT support. Given a ticket description, priority, and impact, decide the assignment group and recommended action."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 5. Create agent with custom prompt and enhanced tools
tools = [suggest_assignment_group, suggest_action, historical_assignment_lookup]
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Load tickets from Excel
file_path = "data_source/ecommerce_inc_list_single.xlsx"
df = pd.read_excel(file_path)

# 7. Run agent for each ticket and collect results
results = []
for idx, row in df.iterrows():
    input_text = f"Ticket: {row['Short Description']}. Priority: {row['Priority']}. Impact: {row['Impact']}"
    response = agent_executor.invoke({"input": input_text})
    results.append({
        "INC Number": row['INC Number'],
        "Short Description": row['Short Description'],
        "Priority": row['Priority'],
        "Impact": row['Impact'],
        "Triage Suggestion": response["output"]
    })

# 8. Save results to new Excel file
result_df = pd.DataFrame(results)
result_df.to_excel("triage_suggestions_output.xlsx", index=False)

print("\nTriage processing complete. Results saved to 'triage_suggestions_output.xlsx'.")
