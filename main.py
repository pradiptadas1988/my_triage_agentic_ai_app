# from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# import pandas as pd

load_dotenv()

# 1. Define tools
@tool
def suggest_assignment_group(description: str) -> str:
    """Suggests the assignment group based on incident description."""
    if "email" in description.lower():
        return "Email Support"
    elif "vpn" in description.lower():
        return "Network Team"
    elif "form" in description.lower():
        return "App Dev"
    else:
        return "IT Support"

@tool
def suggest_action(priority: str, impact: str) -> str:
    """Suggests next triage action based on priority and impact."""
    if priority.lower() in ["high", "critical"] or impact.lower() == "high":
        return "Escalate to Incident Manager"
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

# 4. Load data (mockup for one ticket)
ticket = {
    "short_description": "Unable to access company VPN from home",
    "priority": "Medium",
    "impact": "High"
}

# 5. Run agent
response = agent_executor.invoke({
    "input": f"Ticket: {ticket['short_description']}. Priority: {ticket['priority']}. Impact: {ticket['impact']}"
})

print("\n--- Agent Suggestion ---\n")
print(response["output"])
