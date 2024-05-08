"""
Tool calling allows a model to detect when one or more tools should be called and respond with the inputs that should be
passed to those tools.
Thoughts:
- This is a different way of achieving the same results as with AgentExecutor

Tasks:
- Try tool_calling for agents
# TOOL CALLING -  Understand when to use tool calling vs langgraph agents : https://github.com/langchain-ai/langchain/blob/master/cookbook/tool_call_messages.ipynb?ref=blog.langchain.dev
# TOOL CALLING - In agents and langgraph --> https://github.com/langchain-ai/langchain/blob/master/cookbook/tool_call_messages.ipynb?ref=blog.langchain.dev
# TOOL CALLING - IN LLMS : https://python.langchain.com/docs/modules/model_io/chat/function_calling/?ref=blog.langchain.dev
# TOOL CALLING - AGENT - https://python.langchain.com/docs/modules/agents/agent_types/tool_calling/?ref=blog.langchain.dev

Bugs:
- bind functions is not working as expected
- tool_calls is empty for first example
"""

from langchain_core.tools import tool
import lang_models as models
from conf import configs

configs_chat = configs.ChatbotConfigs(embeddings_deployment="text-embedding-ada-002",
                                      llm_deployment="langchain_model",
                                      llm_type='azure_chat_openai',
                                      # 'hugging_face'  # 'azure_openai', 'azure_chat_openai'
                                      is_authentication_for_prod=True,
                                      saver_version=False,
                                      pg_collection_name='emb_metadata_docs_patent',  # 'test_emb_metadata'
                                      blob_container_name='patent-applications-container'
                                      # "policies-container" #'patent-applications'
                                      )


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

llm = models.get_llm_instance(configs_chat)

llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"

# res = llm_with_tools.invoke(query)
#
# print(res.tool_calls)

# # ------------------------------
# # APPROACH 2
# # FROM https://github.com/langchain-ai/langchain/issues/15012
# # ------------------------------
#
# # llm with tool
# # Create tool
#
#
# from langchain.tools.render import format_tool_to_openai_function
#
# llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
# query = "What is 3 * 12? Also, what is 11 + 49?"
#
# print(llm_with_tools.invoke(query))


# ------------------
# TOOL CALLING AGENT
# ------------------
from agent_tools import initialize_tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

duo_search = initialize_tools(['duo_duck_search'])
duo_search = duo_search[0]
from langchain_experimental.tools import PythonAstREPLTool


tools = [PythonAstREPLTool()]
# tools = [add, multiply]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can execute requests by the user in sequence,"
            " following the steps provided by the user.Make sure to use the PythonAstREPLTool tool to execute code.", # Make sure to use the duo_duck_search tool for information.
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

query = """print the glimpse of iris.csv.
Preprocess the data before training the model.
Train a random forest model to predict the Species column.
Print the first 5 rows of iris.csv.
Save the classification report in result.txt inside directory data """

ans = agent_executor.invoke({"input": query})

print(ans)



