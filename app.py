import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# =========================================
# 1. LOAD ENVIRONMENT VARIABLES & PAGE SETUP
# =========================================
os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

st.set_page_config(
    page_title="LangChain Weather & Search Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Research & Weather Agent")
st.caption("Powered by GPT-4o-mini, Tavily Search, and Weatherstack")

# =========================================
# 2. SIDEBAR CONTROLS (NO API KEYS)
# =========================================
with st.sidebar:
    st.header("🛠️ Available Tools")
    st.markdown(
        "- **Tavily Search**: Real-time web & news search\n"
        "- **Weatherstack**: Live city weather data"
    )
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Verify keys exist in .env without exposing them
if not all([OPENAI_API_KEY, TAVILY_API_KEY, WEATHERSTACK_API_KEY]):
    st.error("Missing API keys! Please ensure OPENAI_API_KEY, TAVILY_API_KEY, and WEATHERSTACK_API_KEY are set in your `.env` file.")
    st.stop()

# =========================================
# 3. DEFINE TOOLS & CACHED AGENT
# =========================================
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    url = f"https://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Error connecting to weather service: {str(e)}"


@st.cache_resource(show_spinner="Initializing Agent...")
def initialize_agent():
    """Initializes and caches the LangChain ReAct agent."""
    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool, get_weather_data]

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

# =========================================
# 4. CHAT INTERFACE
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me to look up recent news, find facts, or check the weather anywhere (e.g., *'Find the capital of India and then find its current weather'*)."
        }
    ]

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "steps" in msg and msg["steps"]:
            with st.expander("🔍 View Agent Thought Process", expanded=False):
                for action, observation in msg["steps"]:
                    st.markdown(f"**Tool Used:** `{action.tool}`")
                    st.markdown(f"**Tool Input:** `{action.tool_input}`")
                    st.markdown(f"**Observation:**\n```text\n{observation}\n```")
                    st.divider()
        st.markdown(msg["content"])

# Handle new user input
if user_query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and calling tools..."):
            try:
                agent_executor = initialize_agent()
                response = agent_executor.invoke({"input": user_query})
                
                output_text = response["output"]
                steps = response.get("intermediate_steps", [])

                if steps:
                    with st.expander("🔍 View Agent Thought Process", expanded=False):
                        for action, observation in steps:
                            st.markdown(f"**Tool Used:** `{action.tool}`")
                            st.markdown(f"**Tool Input:** `{action.tool_input}`")
                            st.markdown(f"**Observation:**\n```text\n{observation}\n```")
                            st.divider()

                st.markdown(output_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output_text,
                    "steps": steps
                })

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")