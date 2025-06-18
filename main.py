from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd

load_dotenv()

# 1. Define tools
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

# 2. Define LLM
llm = ChatOpenAI(model="gpt-4o")

# 3. Define custom prompt with agent_scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a triage assistant for IT support. Given a ticket description, priority, and impact, decide the assignment group and recommended action."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. Create agent with custom prompt
tools = [suggest_assignment_group, suggest_action]
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Load tickets from Excel
file_path = "data_source/ecommerce_inc_list.xlsx"
df = pd.read_excel(file_path)

# 6. Run agent for each ticket and collect results
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

# 7. Save results to new Excel file
result_df = pd.DataFrame(results)
result_df.to_excel("triage_suggestions_output.xlsx", index=False)

print("\nTriage processing complete. Results saved to 'triage_suggestions_output.xlsx'.")
