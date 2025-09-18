import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process,LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from memory import (
    get_user_profile, set_user_profile,
    retrieve_context, store_conversation, update_profile
)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize LLM and tools
llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0,
    api_key=api_key,
)

print()
# Initialize the DuckDuckGo tool
duckduckgo_search = DuckDuckGoSearchRun()


# Use the @tool decorator to wrap the search function for CrewAI
@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """
    Search the web for up-to-date information on a given query.
    This tool is useful for finding product details, reviews, and specifications.
    """
    return duckduckgo_search.run(query)


def crewai_chat(user_id: str, query: str):
    """
    Orchestrates a CrewAI team to provide personalized product recommendations.

    Args:
        user_id (str): The unique identifier for the user.
        query (str): The user's product query.

    Returns:
        str: A summary of the final personalized recommendations.
    """
    # 1. Retrieve User Profile and Context
    profile = get_user_profile(user_id)
    if not profile:
        profile = {"budget": "under $500", "style": "modern", "brands": ["Sony", "Apple"]}
        set_user_profile(user_id, profile)

    history = retrieve_context(user_id, query)

    # 2. Define the Agents
    # Each agent has a specific role, goal, and backstory.
    # The 'tools' list must contain CrewAI-wrapped tools.
    research_agent = Agent(
        role="Product Researcher",
        goal="Find detailed product information from the web",
        backstory="An expert in online shopping and product specifications, "
                  "focused on finding accurate and comprehensive data.",
        tools=[web_search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False  # This agent does not delegate tasks
    )

    analyst_agent = Agent(
        role="Product Analyst",
        goal="Analyze search results and compare products based on user preferences",
        backstory="A meticulous specialist who compares products, features, and prices "
                  "to match them with the user's budget, style, and brand preferences.",
        llm=llm,
        verbose=True
    )

    recommender_agent = Agent(
        role="Personal Recommender",
        goal="Summarize and present the best personalized recommendations",
        backstory="A friendly and loyal shopping assistant who remembers past interactions "
                  "and provides a concise, final list of recommended products.",
        llm=llm,
        verbose=True
    )

    # 3. Define the Tasks
    # Tasks are the specific actions for each agent.
    # The 'context' parameter links the output of one task to the input of another.
    research_task = Task(
        description=f"Search the web for the user's query: '{query}'. "
                    f"Consider the user's profile: {profile} to refine the search. "
                    "Focus on product reviews, tech specs, and pricing.",
        expected_output="A list of raw search results and key product data.",
        agent=research_agent
    )

    analysis_task = Task(
        description=f"Analyze the research results provided. "
                    f"Compare the products based on the user's profile: {profile}. "
                    "Create a detailed comparison of 3-5 top products, highlighting "
                    "their pros and cons relative to the user's preferences.",
        expected_output="A structured comparison of 3-5 products.",
        agent=analyst_agent,
        context=[research_task]
    )

    recommendation_task = Task(
        description=f"Synthesize the product analysis and the past conversation history: {history}. "
                    "Craft a final, personalized recommendation. "
                    "Present the recommendations in a clear and friendly format, "
                    "explaining why each product is a good fit for the user.",
        expected_output="A final list of personalized product recommendations.",
        agent=recommender_agent,
        context=[analysis_task]
    )

    # 4. Create and Run the Crew
    # The 'process=Process.sequential' ensures tasks run in order.
    crew = Crew(
        agents=[research_agent, analyst_agent, recommender_agent],
        tasks=[research_task, analysis_task, recommendation_task],
        process=Process.sequential,
        manager_llm=llm,  # CrewAI V0.33.x and above require a manager_llm
        verbose=True  # Verbose level 2 for detailed output
    )

    result = crew.kickoff(inputs={"query": query})

    # 5. Store Conversation and Update Profile
    store_conversation(user_id, query, str(result))
    update_profile(user_id, f"User searched for {query}")

    return result