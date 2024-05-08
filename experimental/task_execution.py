"""
ADD INFO HERE
"""

from langchain_experimental.tools import PythonAstREPLTool
from colorama import Fore
import re
import lang_models as lm_models
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

llm = lm_models.get_llm_instance(configs_chat)


class PythonREPL:
    def __init__(self):
        self.local_vars = {}
        self.python_tool = PythonAstREPLTool()

    def run(self, code: str) -> str:
        output = str(self.python_tool.run(code))
        if output == "":
            return "Your code is executed successfully"
        else:
            return output


python_repl = PythonREPL()


def get_tool_info(tool_name):
    tools = {
        "python": {"type": "tool",
                   "name": "python",
                   "use": "Use this to execute python code. Display your results using the print function.",
                   "input": "Input should be a valid python code. Ensure proper indentation",
                   "function": python_repl},
    }
    return tools[tool_name]


tools = []
value_dict = {}
tools_description = "\n\nYou can use the following actions:\n\n"
choice = 'python'
tools.append(choice)
tool_info = get_tool_info(choice)
tools_description = tools_description + "Action Name: " + tool_info["name"] + "\nWhen To Use: " + tool_info[
    "use"] + "\nInput: " + tool_info["input"]
tools_description = tools_description + "\n\n"
value_dict[choice] = tool_info["function"]


def run(content, instruction):
    content = content.replace('<<instruction>>', instruction)

    print('content', content)
    count = 0
    while (True):
        count = count + 1
        if count > 10:
            raise ValueError("Too many steps")

        # print(Fore.BLUE + content)
        output = llm.predict(content) # TODO: REPLACE THIS WITH AN AGENT
        output = output.replace("\nObservation:", "")

        print(Fore.MAGENTA + output)
        regex = r"Action\s*\d*\s*:(.*?)\nInput\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, output, re.DOTALL)

        if "Final Answer:" in output and not match:
            break
            # print(output.split("Final Answer:")[1])

        if "Step" not in output:
            print(Fore.YELLOW + "The model didnt output a step.")
            output = "Please follow the format Step/Reason/Action/Input/Observation"
            content = content + "\n" + output
            continue

        if "Reason" not in output:
            print(Fore.YELLOW + "The model didnt output a reason.")
            output = "Please follow the format Step/Reason/Action/Input/Observation"
            messages = content + "\n" + output
            continue

        if output.count("Input") > 1:
            print(Fore.YELLOW + "The model went crazy.")
            output = "Please go one step at a time."
            content = content + "\n" + output
            continue

        if not match:
            print(Fore.RED + "The model was sidetracked.")
            output = "You are not following the format. Please follow the given format."
            messages = [{"role": "user", "content": content + "\n" + output}]
            continue

        action = match.group(1).strip().lower()
        if action not in tools:
            output = f"Invalid Action. Your action should be one of {tools}."
            print(Fore.YELLOW + "The agent forgot his tools." + output)
            content = content + "\n" + output
            continue

        action_input = match.group(2)
        match = re.search(r"Step (\d+): (.*)", output)
        step_number = int(match.group(1)) + 1
        observation = value_dict[action](action_input)
        print(Fore.GREEN + "\nObservation: " + str(observation))
        output = output + "\nObservation: " + str(observation)
        content = content + "\n" + output


content = """You can use the following actions:

Action Name: python When To Use: When you send a message containing Python code to python, it will be executed in a
stateful Jupyter notebook environment. python will respond with the output of the execution seconds. Input: Input
should be a valid python code. Ensure proper indentation

Acomplish the task in steps. If you get error in previous step fix it in the current step. Use the following format:

Step 1: The first step
Reason: Reason for taking this step
Action: the action to take, should be one of ['python'].
Input: the input to the action
Observation: the result of the action

Step 2: The second step
Reason: Reason for taking this step
Action: the action to take, should be one of ['python'].
Input: the input to the action
Observation: the result of the action

... (this Step/Reason/Action/Input/Observation repeats for all steps)

Once you have completed all the steps, your final answer should be in the format:
Final Answer: I have completed all the steps

Begin

<<instruction>>
"""


def extract_code_from_block(response):
    if '```' not in response:
        return response
    if '```python' in response:
        code_regex = r'```python(.+?)```'
    else:
        code_regex = r'```(.+?)```'
    code_matches = re.findall(code_regex, response, re.DOTALL)
    code_matches = [item for item in code_matches]
    return "\n".join(code_matches)


def fix_error(code, result):
    count = 0
    while "Your code has the following error." in result:
        error = result.replace("Your code has the following error. Please provide the corrected code.", "")
        user_input = f"""Here is the Code and the Error.

        Code:
        {code}
        Error:
        {error}

        Fix the given using the following format:
        Explanation: Explain the error in the code

        Corrected Code: Put the Corrected Code here"""

        print(Fore.RED + "Code needs some correction.\n")
        code = llm.predict(user_input)
        code = code[code.rfind('Corrected Code:') + len("Corrected Code:"):]
        code = extract_code_from_block(code)
        print(Fore.CYAN + "Corrected Code.\n" + code)
        result = python_repl.run(code)
        print(Fore.BLUE + "Result.\n" + result)
        count += 1
        if count > 5:
            raise ValueError("Too many steps")
    print(Fore.GREEN + "Code has been corrected.\n" + code)
    return result


def execute(instruction):
    return run(content, instruction)


command = """print the glimpse of iris.csv.
Preprocess the data before training the model.
Train a random forest model to predict the Species column.
Print the first 5 rows of iris.csv.
Save the classification report in result.txt inside directory data """

execute(command)
