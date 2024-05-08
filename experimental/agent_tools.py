import os
from typing import Any, List, Optional

import pandas as pd
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, ZeroShotAgent, create_react_agent
from langchain.chains import ConversationChain, LLMChain
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLLM
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

import custom_prompts.coder_bot
from lang_models import BaseChain
from langchain.prompts.prompt import PromptTemplate

load_dotenv()

os.getenv('GOOGLE_CSE_ID')
os.getenv('GOOGLE_API_KEY')


def get_custom_tool(func: Any, name: str, description: str) -> Tool:
    """
    Create a custom tool with the provided function, name, and description.

    Args:
        func (Any): The function to be associated with the tool.
        name (str): The name of the tool.
        description (str): The description of the tool.

    Returns:
        Tool: The custom tool object.
    """
    return Tool(
        name=name,
        func=func.run,
        description=description
    )


def initialize_agent_executor(language_model: AzureChatOpenAI, tools: List[Tool], memory: Any = None,
                              custom_template: Any = None) -> AgentExecutor:
    """
    Initialize the agent executor with the provided language model, tools, memory, and custom template.

    Args:
        language_model (AzureChatOpenAI): The language model to be used.
        tools (List[Tool]): The list of tools to be used.
        memory (Any, optional): The memory to be used. Defaults to None.
        custom_template (Any, optional): The custom template to be used. Defaults to None.

    Returns:
        AgentExecutor: The initialized agent executor.
    """
    prompt = hub.pull("hwchase17/react")

    if custom_template is not None:
        prompt.template = custom_template

    # TODO: IMPLEMENT DIFFERENT AGENTS
    # llm_with_stop = language_model.bind(stop=["\nFinal Answer"])  # TEST
    agent = create_react_agent(language_model, tools, prompt)
    return AgentExecutor(agent=agent,
                         tools=tools,
                         verbose=True,  # True
                         handle_parsing_errors='Return Final Answer',
                         # "Check your output and make sure it conforms, use the Action/Action Input syntax",
                         # True,
                         memory=memory,
                         # max_execution_time=30,
                         max_iterations=15,
                         early_stopping_method='generate'  #
                         )


# TODO MEMORY ADDED , CHECK WITH OTHER FUNCTIONS
def load_llms_chain_as_tools(llm: Any, template, memory=None) -> ConversationChain:
    """
    Load the tool functions using the provided language model.

    Args:
        llm (Any): The language model to be used.

    Returns:
        ConversationChain: The initialized conversation chain.
    """

    answer_prompt = PromptTemplate(input_variables=["history", "input"],
                                   template=template)
    if memory is None:
        code_assistant = ConversationChain(
            llm=llm,
            prompt=answer_prompt
        )
    else:
        code_assistant = ConversationChain(
            llm=llm,
            prompt=answer_prompt,
            memory=memory
        )

    code_assistant_tool = BaseChain(code_assistant, chain_type='ConversationChain')

    return code_assistant_tool


def initialize_tools(tools_names_to_use: List[str], language_model=None, rag_chain_tool=None
                     ) -> List:
    """
    Initialize tools based on the specified tool names and configurations.

    Args:
        tools_names_to_use: A list of tool names to initialize.
        language_model: The language model chain.
        rag_chain_tool: The RAG (Retrieval-Augmented Generation) chain tool.
        llm_math_chain: The language model math chain.

    Returns:
        A list of initialized tools.
    """
    tools = list()

    for tool_name in tools_names_to_use:

        if tool_name == 'se_language_model' and rag_chain_tool is not None:
            rag_bot = get_custom_tool(func=rag_chain_tool,
                                      name="Stora Enso Large Language Model",
                                      description="Use this tool to find internal legal information specific to Stora "
                                                  "Enso, which may not be publicly available on the internet."

                                      )
            tools.append(rag_bot)
        elif tool_name == 'duo_duck_search':
            duo_duck_search = DuckDuckGoSearchRun()
            duo_duck_search = get_custom_tool(func=duo_duck_search,
                                              name="duo_duck_search",
                                              description="useful for searching the internet for current or even"
                                                          "future information"

                                              )
            tools.append(duo_duck_search)

        # elif tool_name == 'math_tool':
        #     problem_chain = LLMMathChain.from_llm(llm=language_model)
        #
        #     math_tool = get_custom_tool(func=problem_chain,
        #                                 name="math_tool",
        #                                 description="Useful for when you need to answer questions about math. "
        #                                             "This tool is only for math questions and nothing else. "
        #                                             "Only input math expressions."
        #                                 )
        #
        #     tools.append(math_tool)

        # elif tool_name == 'python_repl':
        #     python_repl = get_custom_tool(func=PythonREPL(),
        #                                   name="python_repl",
        #                                   description="A Python shell. Use this to execute python "
        #                                               "commands. Input"
        #                                               "should be a valid python command. If you want to see the output "
        #                                               "of a value, you should print it out with `print(...)`."
        #                                   )
        #     tools.append(python_repl)

        # # TESTING
        elif tool_name == 'python_repl':
            python_repl = get_custom_tool(func=PythonAstREPLTool(),
                                          name="python_repl",
                                          description="A Python shell. Use this to execute python "
                                                      "commands. Input"
                                                      "should be a valid python command. If you want to see the output "
                                                      "of a value, you should print it out with `print(...)`."
                                          )
            tools.append(python_repl)

        elif tool_name == 'programmer' and language_model is not None:
            code_assistant_func = load_llms_chain_as_tools(language_model,
                                                           custom_prompts.coder_bot.agent_template_code_programmer)
            code_assistant = get_custom_tool(func=code_assistant_func,
                                             name="programmer",
                                             description="An LLM and software engineer with expertise in tackling "
                                                         "intricate Python code and refining software architecture using "
                                                         "design pattern principles. Not useful for generating plots."
                                             )
            tools.append(code_assistant)

        # elif tool_name == 'code_executer' and language_model is not None:
        #     code_exec_func = load_llms_chain_as_tools(language_model,
        #                                                    custom_prompts.agent_template_code_executer)
        #     code_assistant = get_custom_tool(func=code_exec_func,
        #                                      name="code_executer",
        #                                      description="An LLM and software engineer with expertise in tackling "
        #                                                  "intricate Python code and refining software architecture using "
        #                                                  "design pattern principles. Not useful for generating plots."
        #                                      )
        #     tools.append(code_assistant)

        elif tool_name == 'plot_displayer' and language_model is not None:
            plot_displayer_func = load_llms_chain_as_tools(language_model, custom_prompts.agent_template_plot_displayer)
            code_assistant = get_custom_tool(func=plot_displayer_func,
                                             name="plot_displayer",
                                             description="Utilize this tool to request the generation of visualizations and plots."
                                             )
            tools.append(code_assistant)

    return tools


def create_pandas_dataframe_agent_custom(
        llm: BaseLLM,
        df: Any,
        memory: Any = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        input_variables: Optional[List[str]] = None,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        max_iterations: Optional[int] = 20,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        **kwargs: Any,
) -> AgentExecutor:
    """
        Construct a pandas agent from a language model and dataframe.

        Args:
            llm (BaseLLM): The language model.
            df (Any): The pandas DataFrame object.
            memory (Any, optional): The memory object. Defaults to None.
            callback_manager (Optional[BaseCallbackManager], optional): The callback manager. Defaults to None.
            input_variables (Optional[List[str]], optional): The input variables. Defaults to None.
            verbose (bool, optional): Verbosity flag. Defaults to False.
            return_intermediate_steps (bool, optional): Flag to return intermediate steps. Defaults to False.
            max_iterations (Optional[int], optional): Maximum number of iterations. Defaults to 20.
            max_execution_time (Optional[float], optional): Maximum execution time. Defaults to None.
            early_stopping_method (str, optional): Early stopping method. Defaults to "force".
            **kwargs (Any): Additional keyword arguments.

        Returns:
            AgentExecutor: The initialized agent executor.
        """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas object, got {type(df)}")
    if input_variables is None:
        input_variables = ["df", "input", "agent_scratchpad"]

    tools = [PythonAstREPLTool(locals={"df": df})]
    # # ---------------------------------------
    # # Experimental
    # plot_displayer = initialize_tools(tools_names_to_use=['plot_displayer'], language_model=llm)
    # tools.extend(plot_displayer)
    # #print('tool_math', tool_math)
    # #print('tools', tools[1])  # TEST
    # # ---------------------------------------

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=custom_prompts.coder_bot.PREFIX_PANDAS,
        suffix=custom_prompts.coder_bot.SUFFIX_PANDAS,
        input_variables=["df", "input", "chat_history", "agent_scratchpad"]
    )

    partial_prompt = prompt.partial(df=str(df.head()))

    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )

    tool_names = [tool.name for tool in tools]

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
        **kwargs,
    )

    # TODO:TRY ONLY AGENT EXECUTOR
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        callback_manager=callback_manager,
        memory=memory,
        handle_parsing_errors=True  # TESTING
    )

# ---------------------------
# APPENDIX
# ---------------------------
