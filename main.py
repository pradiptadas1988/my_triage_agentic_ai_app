from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd

load_dotenv()

# 1. Define tools (updated for e-commerce incidents)
@tool
def suggest_assignment_group(description: str) -> str:
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

@tool
def suggest_action(priority: str, impact: str) -> str:
    """Suggests next triage action based on priority and impact."""
    p = priority.lower()
    i = impact.lower()
    if p == "critical" or i == "high":
        return "Escalate to Incident Manager"
    elif p == "high" or i == "medium":
        return "Immediate attention by team lead"
    else:
        return "Assign to appropriate group and monitor"

# 2. Load historical data for similarity-based matching
historical_df = pd.read_excel("data_source/historical_incidents.xlsx")

@tool
def historical_assignment_lookup(description: str) -> str:
    """Looks up the best assignment group from historical incidents using token overlap."""
    desc_tokens = set(description.lower().split())
    best_group = "General IT Support"
    max_overlap = 0

    for _, row_data in historical_df.iterrows():
        hist_desc = str(row_data.get("Short Description", "")).lower()
        hist_tokens = set(hist_desc.split())
        overlap = len(desc_tokens & hist_tokens)
        if overlap > max_overlap:
            max_overlap = overlap
            best_group = row_data.get("Assignment Group", best_group)

    return best_group

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
