"""
ADD INFO HERE
Status:
- There is still a validation error

"""


# LEETCODE: https://github.com/anurag899/openAI-project/blob/main/Multi-Agent%20Coding%20Framework%20using%20LangGraph/LangGraph%20-%20Code%20Development%20using%20Multi-Agent%20Flow.ipynb 40 MIN

import os
from langchain.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_agent_executor
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
import operator

# -------
# HELPERS
# -------
from conf import configs

import lang_models as comps

configs_chat = configs.ChatbotConfigs(embeddings_deployment="text-embedding-ada-002",
                                      llm_deployment="langchain_model",
                                      llm_type='azure_chat_openai',  # 'hugging_face'  # 'azure_openai'
                                      is_authentication_for_prod=True,
                                      saver_version=False,
                                      pg_collection_name='emb_metadata_docs_patent',  # 'test_emb_metadata'
                                      blob_container_name='patent-applications-container'
                                      # "policies-container" #'patent-applications'
                                      )
llm = comps.get_llm_instance(configs_chat)


# -------
# CODER
# -------
class Code(BaseModel):
    """Plan to follow in future"""

    code: str = Field(
        description="Detailed optmized error-free Python code on the provided requirements"
    )


from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate

code_gen_prompt = ChatPromptTemplate.from_template(
    '''**Role**: You are a expert software python programmer. You need to develop python code

**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
efficient, readable, and well-commented.

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient way.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code

*REQUIREMENT*
{requirement}'''
)

coder = create_structured_output_runnable(Code, llm, code_gen_prompt)

# print(coder.invoke({'requirement':'generate a python code'}))

code_ = coder.invoke({'requirement': 'Generate fibbinaco series'})


# --------
# TESTER
# --------

class Test(BaseModel):
    """Plan to follow in future"""

    Input: List[List] = Field(
        description="Input for Test cases to evaluate the provided code"
    )
    Output: List[List] = Field(
        description="Expected Output for Test cases to evaluate the provided code"
    )


from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate

test_gen_prompt = ChatPromptTemplate.from_template(
    '''**Role**: As a tester, your task is to create Basic and Simple test cases based on provided Requirement and Python Code.
These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.
**1. Basic Test Cases**:
- **Objective**: Basic and Small scale test cases to validate basic functioning
**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.
**Instructions**:
- Implement a comprehensive set of test cases based on requirements.
- Pay special attention to edge cases as they often reveal hidden bugs.
- Only Generate Basics and Edge cases which are small
- Avoid generating Large scale and Medium scale test case. Focus only small, basic test-cases
*REQURIEMENT*
{requirement}
**Code**
{code}
'''
)
tester_agent = create_structured_output_runnable(
    Test, llm, test_gen_prompt
)

# # -----------------------------------------------------
# # TEST PART
# from typing import Optional
#
# from langchain.chains import create_structured_output_runnable
# from langchain_openai import ChatOpenAI
# from langchain_core.pydantic_v1 import BaseModel, Field
#
#
# class RecordDog(BaseModel):
#     '''Record some identifying information about a dog.'''
#
#     name: str = Field(..., description="The dog's name")
#     color: str = Field(..., description="The dog's color")
#     fav_food: Optional[str] = Field(None, description="The dog's favorite food")
#
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an extraction algorithm. Please extract every possible instance"),
#         ('human', '{input}')
#     ]
# )
# structured_llm = create_structured_output_runnable(
#     RecordDog,
#     llm,
#     mode="openai-tools",
#     enforce_function_usage=True,
#     return_single=True
# )
# structured_llm.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
#
# # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
#
#
#
# from typing import Optional
#
# from langchain.chains import create_structured_output_runnable
# from langchain_openai import ChatOpenAI
#
#
# dog_schema = {
#     "type": "function",
#     "function": {
#         "name": "record_dog",
#         "description": "Record some identifying information about a dog.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "name": {
#                     "description": "The dog's name",
#                     "type": "string"
#                 },
#                 "color": {
#                     "description": "The dog's color",
#                     "type": "string"
#                 },
#                 "fav_food": {
#                     "description": "The dog's favorite food",
#                     "type": "string"
#                 }
#             },
#             "required": ["name", "color"]
#         }
#     }
# }
#
#
# # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
# structured_llm = create_structured_output_runnable(
#     dog_schema,
#     llm,
#     mode="openai-tools",
#     enforce_function_usage=True,
#     return_single=True
# )
# structured_llm.invoke("Harry was a chubby brown beagle who loved chicken")
