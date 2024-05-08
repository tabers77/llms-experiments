"""
Sources: https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb

- We may need to return a structured JSON object
- Can agent supervisor follow instructions ?

Bugs:
- The method JsonOutputFunctionsParser is still not working with AzureChatOpenAi
"""

# CURRENT ERROR: https://github.com/langchain-ai/langchain/pull/20660
# https://community.openai.com/t/invalid-json-response-ending-with/669624
# BUG WITH AZURE OPEN AI : AttributeError: 'AzureOpenAI' object has no attribute 'bind_functions'


# from typing import Annotated, List, Tuple, Union
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_openai import AzureChatOpenAI
from experimental.output_parsers_custom import JsonOutputFunctionsParser  # CUSTOM FUNCTION
# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

import lang_models as comps
from conf import configs

import operator
from typing import Annotated, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from agent_tools import initialize_tools

# --------
# CONFIGS
# --------
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
llm = comps.get_llm_instance(configs_chat)

# from langchain_core.utils.function_calling import convert_to_openai_function
#
# llm = chat_openai_instance = convert_to_openai_function(comps.get_llm_type(configs_chat))

duo_search = initialize_tools(['duo_duck_search'])
duo_search = duo_search[0]

# tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()


# ************************** EXAMPLE 1 **************************
# --------
# HELPERS
# --------
def create_agent(llm: AzureChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)

    executor = AgentExecutor(agent=agent,
                             tools=tools,
                             handle_parsing_errors=True,
                             verbose=True,
                             return_intermediate_steps=True)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# -----------------------
# CREATE AGENT SUPERVISOR
# -----------------------

members = ["Researcher", "Coder"]


system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
)


# ---------------
# CONSTRUCT GRAPH
# ---------------

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


# ------
# AGENTS
# ------
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from typing import Annotated, List, Tuple, Union


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


research_agent = create_agent(
    llm,
    [scrape_webpages],
    "You are a research assistant who can scrape specified urls for more detailed information using the scrape_webpages function.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

#
# search_agent = create_agent(llm,
#                             [duo_search],
#                             "You are a research assistant who can search for up-to-date info"
#                             )
# search_node = functools.partial(agent_node, agent=search_agent, name="Search")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
# workflow.add_node("Search", search_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# print(graph)
# print(dir(graph))

# ---------------
# INVOKE TEAM
# ---------------

query1 = "Code hello world and print it to the terminal and Write a brief research report on the North American sturgeon."
query2 = "Write a brief research report on the North American sturgeon."
query3 = "Create Python code intentionally containing a bug. Research how to fix the bug, and then create the bug-free code"
query4 = "Write an advanced research paper about the future role of Markov chains in LLM reasoning using information from  this research paper:https://arxiv.org/pdf/2305.14992 . Later, write a Python script on how we could implement Markov chains reasoning in LLMs."
query5 = "when is Taylor Swift's next tour in 2024?"
# for s in graph.stream(
#         {
#             "messages": [
#                 HumanMessage(content=query4)
#             ]
#         }
# ):
#     if "__end__" not in s:
#         print('End reached')
#         keys_list = list(s.keys())
#
#         # Convert list of keys to a string
#         current_step = ', '.join(keys_list)
#         try:
#             print('content', s[current_step]['messages'][0].content)
#         except KeyError:
#             pass
#         print("----")


# for s in graph.stream(
#         {
#             "messages": [HumanMessage(content="Write a brief research report on the North American sturgeon.")]},
#         {"recursion_limit": 150},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("---")

for s in graph.stream(
        {"messages": [HumanMessage(content=query4)]},
        {"recursion_limit": 100},
):
    if "__end__" not in s:
        print(s)
        print("----")


# ************************** EXAMPLE 2 **************************
#
#
# # https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/multi-agent-collaboration.ipynb
#
# from dotenv import load_dotenv
# load_dotenv()
# import json
#
# from langchain_core.messages import (
#     AIMessage,
#     BaseMessage,
#     ChatMessage,
#     FunctionMessage,
#     HumanMessage,
# )
# from langchain.tools.render import format_tool_to_openai_function
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.graph import END, StateGraph
# from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
#
#
# def create_agent(llm, tools, system_message: str):
#     """Create an agent."""
#     functions = [format_tool_to_openai_function(t) for t in tools]
#
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a helpful AI assistant, collaborating with other assistants."
#                 " Use the provided tools to progress towards answering the question."
#                 " If you are unable to fully answer, that's OK, another assistant with different tools "
#                 " will help where you left off. Execute what you can to make progress."
#                 " If you or any of the other assistants have the final answer or deliverable,"
#                 " prefix your response with FINAL ANSWER so the team knows to stop."
#                 " You have access to the following tools: {tool_names}.\n{system_message}",
#             ),
#             MessagesPlaceholder(variable_name="messages"),
#         ]
#     )
#     prompt = prompt.partial(system_message=system_message)
#     prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
#     return prompt | llm.bind_functions(functions)
#
#
# # -------------
# # DEFINE TOOLS
# # -------------
#
# from langchain_core.tools import tool
# from typing import Annotated
# from langchain_experimental.utilities import PythonREPL
# # from langchain_community.tools.tavily_search import TavilySearchResults
# #
# # tavily_tool = TavilySearchResults(max_results=5)
#
# # Warning: This executes code locally, which can be unsafe when not sandboxed
#
# repl = PythonREPL()
#
#
# @tool
# def python_repl(
#         code: Annotated[str, "The python code to execute to generate your chart."]
# ):
#     """Use this to execute python code. If you want to see the output of a value,
#     you should print it out with `print(...)`. This is visible to the user."""
#     try:
#         result = repl.run(code)
#     except BaseException as e:
#         return f"Failed to execute. Error: {repr(e)}"
#     return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"
#
#
# # -------------
# # CREATE A GRAPH
# # -------------
#
# import operator
# from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
#
# from langchain.agents import create_openai_functions_agent
# from langchain.tools.render import format_tool_to_openai_function
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#
# from langchain_openai import ChatOpenAI
# from typing_extensions import TypedDict
#
#
# # This defines the object that is passed between each node
# # in the graph. We will create different nodes for each agent and tool
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     sender: str
#
#
# import functools
#
#
# # Helper function to create a node for a given agent
# def agent_node(state, agent, name):
#     result = agent.invoke(state)
#     # We convert the agent output into a format that is suitable to append to the global state
#     if isinstance(result, FunctionMessage):
#         pass
#     else:
#         result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
#     return {
#         "messages": [result],
#         # Since we have a strict workflow, we can
#         # track the sender so we know who to pass to next.
#         "sender": name,
#     }
#
# #llm = ChatOpenAI(model="gpt-4-1106-preview")
# from langchain_community.tools.tavily_search import TavilySearchResults
#
# tavily_tool = TavilySearchResults(max_results=5)
#
# # Research agent and node
# research_agent = create_agent(
#     llm,
#     [tavily_tool],
#     system_message="You should provide accurate data for the chart generator to use.",
# )
# research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
#
# # Chart Generator
# chart_agent = create_agent(
#     llm,
#     [python_repl],
#     system_message="Any charts you display will be visible by the user.",
# )
# chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart Generator")
#
# tools = [tavily_tool, python_repl]
# tool_executor = ToolExecutor(tools)
#
#
# def tool_node(state):
#     """This runs tools in the graph
#
#     It takes in an agent action and calls that tool and returns the result."""
#     messages = state["messages"]
#     # Based on the continue condition
#     # we know the last message involves a function call
#     last_message = messages[-1]
#     # We construct an ToolInvocation from the function_call
#
#     print('JSON LOADS ARGUMENTS:', last_message.additional_kwargs["function_call"]["arguments"])
#     tool_input = json.loads(
#         last_message.additional_kwargs["function_call"]["arguments"]
#     )
#     print('tool_input', tool_input)
#
#     # We can pass single-arg inputs by value
#     if len(tool_input) == 1 and "__arg1" in tool_input:
#         tool_input = next(iter(tool_input.values()))
#     tool_name = last_message.additional_kwargs["function_call"]["name"]
#     action = ToolInvocation(
#         tool=tool_name,
#         tool_input=tool_input,
#     )
#     # We call the tool_executor and get back a response
#     response = tool_executor.invoke(action)
#     # We use the response to create a FunctionMessage
#     function_message = FunctionMessage(
#         content=f"{tool_name} response: {str(response)}", name=action.tool
#     )
#     # We return a list, because this will get added to the existing list
#     return {"messages": [function_message]}
#
#
# # Either agent can decide to end
# def router(state):
#     # This is the router
#     messages = state["messages"]
#     last_message = messages[-1]
#     if "function_call" in last_message.additional_kwargs:
#         # The previus agent is invoking a tool
#         return "call_tool"
#     if "FINAL ANSWER" in last_message.content:
#         # Any agent decided the work is done
#         return "end"
#     return "continue"
#
#
# workflow = StateGraph(AgentState)
#
# workflow.add_node("Researcher", research_node)
# workflow.add_node("Chart Generator", chart_node)
# workflow.add_node("call_tool", tool_node)
#
# workflow.add_conditional_edges(
#     "Researcher",
#     router,
#     {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
# )
# workflow.add_conditional_edges(
#     "Chart Generator",
#     router,
#     {"continue": "Researcher", "call_tool": "call_tool", "end": END},
# )
#
# workflow.add_conditional_edges(
#     "call_tool",
#     # Each agent node updates the 'sender' field
#     # the tool calling node does not, meaning
#     # this edge will route back to the original agent
#     # who invoked the tool
#     lambda x: x["sender"],
#     {
#         "Researcher": "Researcher",
#         "Chart Generator": "Chart Generator",
#     },
# )
# workflow.set_entry_point("Researcher")
# graph = workflow.compile()
#
# for s in graph.stream(
#         {
#             "messages": [
#                 HumanMessage(
#                     content="Fetch the UK's GDP over the past 5 years,"
#                             " then draw a line graph of it."
#                             " Once you code it up, finish."
#                 )
#             ],
#         },
#         # Maximum number of steps to take in the graph
#         {"recursion_limit": 150},
# ):
#     print(s)
#     print("----")
