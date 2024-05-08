"""
Sequential chains are particularly useful when you need to make multiple calls to an LLM (Large Language Model) where
the output of one call serves as the input for the next call.

Questions:
- Is it possible to use tools in sequential chains? --> Nope I could create my own sequential object

Next steps:
- Try to achieve a simple version of task execution.
"""
import os.path

import lang_models as l_models
from conf import configs
from langchain.prompts import PromptTemplate

from langchain.chains import SimpleSequentialChain, SequentialChain, LLMChain

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
llm = l_models.get_llm_instance(configs_chat)

from langchain_community.tools import DuckDuckGoSearchRun
import agent_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import custom_prompts

debug_researcher = agent_tools.get_custom_tool(func=DuckDuckGoSearchRun(),
                                               name="debug_researcher",
                                               description="useful for searching the internet for solution to code bugs"

                                               )

# ------------------------
# SEQUENTIAL CHAIN EXAMPLE
# ------------------------

# CHAIN 1
prompt_template_programmer = PromptTemplate(input_variables=["input", "chat_history"],
                                            template=custom_prompts.coder_bot.agent_template_code_generator)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")  # TEST

chain_programmer = LLMChain(llm=llm, prompt=prompt_template_programmer, memory=memory)

prompt_template_analyzer = PromptTemplate(input_variables=["input", "chat_history"],
                                          template=custom_prompts.coder_bot.agent_template_code_analyzer)
chain_analyzer = LLMChain(llm=llm, prompt=prompt_template_analyzer, memory=memory)

# CHAIN 2
prompt_template_tester = PromptTemplate(input_variables=["input"],
                                        template=custom_prompts.coder_bot.agent_template_code_tester)
chain_code_tester = LLMChain(llm=llm, prompt=prompt_template_tester)

# CHAIN 3
prompt_template_final_template = PromptTemplate(input_variables=["input"],
                                                template=custom_prompts.coder_bot.agent_template_final_template)
chain_three = LLMChain(llm=llm, prompt=prompt_template_final_template)

memory_chit_chat = l_models.CustomConversationBufferMemory(llm_custom=llm,
                                                           memory_key="history",
                                                           return_messages=True
                                                           )

# Observe that SequentialChain when we have multiple inputs

# overall_chain = SequentialChain(
#     input_variables=["input"],
#     memory=memory_chit_chat,
#     chains=[chain_one, chain_two],
#     verbose=True)

# overall_chain = SimpleSequentialChain(
#     # input_variables=["input"],
#     # memory=memory_chit_chat,
#     chains=[chain_programmer, chain_code_tester],
#     verbose=True)

# ans = overall_chain.run("generate a python code to remove duplicates and a code to generate barplots")
#
# print('ANSWER', ans)


# ----------------------
# CUSTOM STEP EXECUTORS
# ----------------------


from langchain_experimental.tools import PythonAstREPLTool


class StepsExecutors:
    @staticmethod
    def individual_steps_executor(query):
        # 1. break down steps in sequences
        # 2. execute sequences  logically
        steps = query.split('-')[1:]

        code_wrapper = ''
        for step in steps:
            print(f'Running step: {step}')
            code = chain_programmer.invoke(step)['text']

            # CODE VALIDATION
            print('CODE', code)
            code_wrapper += code

            # # -------------------------------------
            # # This should be conditionally
            # code_executer = PythonAstREPLTool()
            # print('EXECUTE', code_executer.run(code))
            #
            # print(f'Finished step: {step}')
            # # -------------------------------------
        print('CODE WRAPPER', code_wrapper)

    @staticmethod
    def general_steps_executor(query):
        code_executer = PythonAstREPLTool()
        debug_query = None
        task_completed = False

        # 1. RESULT GENERATOR
        while task_completed is False:
            # 1. Code generator
            query = query if debug_query is None else debug_query

            code = chain_programmer.invoke(query)['text']
            print('CODE1:', code)

            # 2. Final Code generation
            # This should be conditionally
            execution_task = code_executer.run(code)

            if execution_task != '':
                print('ERROR:', execution_task)
                debug_query = f'This script {code} you generated previously has the following error: {execution_task} ' \
                              f'when trying to execute it using PythonAstREPLTool. Fix it!'

            else:
                task_completed = True

            # 2. ANALYZER


# def generate_prompt(target_variable_name, dataset_path, dest_folder, loss_functions, model_name=None, model_names=None):
#     # #  PROMPT # 1
#     # return f'This is a supervised case, the target variable name is {target_variable_name}. You will be using the dataset located at {dataset_path}. ' \
#     #        f'1. Create a train-test split. ' \
#     #        f'2. Generate a feature importance graph using {model_name} and save it in the directory {dest_folder}. ' \
#     #        f'3. Train a {model_name} model, evaluate the results using loss function/s {loss_functions} on test data, and save the evaluation results in the directory{dest_folder}'
#
#     # #  PROMPT # 2 - EDA
#     # return f'This is a supervised case, the target variable name is {target_variable_name}. You will be using the dataset located at {dataset_path}. ' \
#     #        f'1. Apply some basic exploratory data analysis (EDA) to the dataset and save the observations in text format in the directory {dest_folder}. ' \
#     #        f'2. Create a train-test split. ' \
#     #        f'3. Generate a feature importance graph using {model_name} and save it in the directory {dest_folder}. ' \
#     #        f'4. Train a {model_name} model, evaluate the results using loss function/s {loss_functions} on test data, and save the evaluation results in the directory{dest_folder} '
#
#     #  PROMPT # 3 - MULTIPLE MODELS
#     return f'This is a supervised case, the target variable name is {target_variable_name}. You will be using the dataset located at {dataset_path}. ' \
#            f'1. EDA: Apply some basic exploratory data analysis (EDA) to the dataset and save the observations in text format in the directory {dest_folder}. ' \
#            f'2. TRAIN TEST SPLIT: Create a train-test split. ' \
#            f'3. FEATURE IMPORTANCE: Generate a feature importance graph using {model_name} and save it in the directory {dest_folder}. ' \
#            f'4. MODEL TRAINING: Train  {model_names} , evaluate/compare the results using loss function/s {loss_functions} on test data, and save the evaluation results in the directory{dest_folder} '
#
# query = generate_prompt(target_variable_name=target_variable_name,
#                         dataset_path=dataset_path,
#                         dest_folder=dest_folder,
#                         loss_functions=loss_functions,
#                         model_name=model_names,
#                         model_names=model_names)

# ---------------------------------------------------------
# INTRO
case_type = 'supervised modelling'
modeling_type = 'regression'
# test_split_threshold = '0.21'

# Example usage:
dataset_path = "../data/dataset.csv"
dest_folder = "../data/temp_files/"

target_variable_name = '"target"'

loss_functions = "mean squared error, mean absolute error"

# model_names = "RandomForestRegressor, LinearRegression"
model_names = "3 different estimators from sklearn and a Stacked model of those 3 estimators (use sklearn.ensemble.StackingRegressor) "

# --------------
# MODELLING CASE
# --------------

# 1. give needed configs/info ----> get unstructured instruction
# There are different cases , we could start with modelling case

# 1. RESULT GENERATOR (text is saved) ---> 2. Extract the text for query   ---> 3. Analyzer (suggestions)....

# -------------------
# 1. RESULT GENERATOR
# -------------------

# base_prompt_modelling_coder = f"This involves {case_type} of {modeling_type}. You'll utilize the dataset located at {dataset_path}. " \
#                         f"If the user requests to save outputs, they should be saved in the directory {dest_folder}. " \
#                         f"The name of the target variable is {target_variable_name}. The loss functions used to " \
#                         f"evaluate the models are {loss_functions}, while the estimators/algorithms employed are " \
#                         f"{model_names}. "

base_prompt_modelling_coder = f"This involves {case_type} of {modeling_type}. You'll utilize the dataset located at {dataset_path}. " \
                              f"If the user requests to save outputs, they should be saved in the directory {dest_folder}. Additionally, the file name for the modeling results should be 'results.txt' . " \
                              f"The name of the target variable is {target_variable_name}. The loss functions used to " \
                              f"evaluate the models are {loss_functions}, while the estimators/algorithms employed are " \
                              f"{model_names}. "

base_prompt_modelling_analyzer = f"This involves {case_type} of {modeling_type}. You'll utilize the dataset located at {dataset_path}. " \
                                 f"If you need to save outputs or results, please ensure they are stored in the directory specified by {dest_folder}. " \
                                 f"The name of the target variable is {target_variable_name}. The loss functions used to " \
                                 f"evaluate the models are {loss_functions}. "

#
# base_prompt_modelling_analyzer = f"You'll utilize the dataset located at {dataset_path}. " \
#                                  f"The modelling should be located in the directory {dest_folder} and the file name for the modeling results should be 'results'. " \

#  1. USER INPUT
user_input = f'1. EDA: Apply some basic exploratory data analysis (EDA) to the dataset and save the observations in text format. ' \
             f'2. TRAIN TEST SPLIT: Create a train-test split. Use test split .30. ' \
             f'3. FEATURE IMPORTANCE: Generate a feature importance graph and save it . ' \
             f'4. MODEL TRAINING: Train models, evaluate/compare the results using loss function/s  on test data, and save results  '


def build_prompt():
    return base_prompt_modelling_coder + 'STEPS TO FOLLOW: ' + user_input


test_query = build_prompt()


# executor = StepsExecutors()
# executor.general_steps_executor(test_query)


# ------------
# 2. ANALYZER
# ------------

def generate_next_steps():
    # Check results in path
    if os.path.exists(f'{dest_folder}/results.txt'):
        # Extract data
        with open(f'{dest_folder}/results.txt', 'r') as f:
            results = f.read()
    else:
        raise ValueError('Path does not exist..')

    recommendations = chain_analyzer.invoke(f'{base_prompt_modelling_analyzer}. Here are the results: {results}')[
        'text']

    with open(f'{dest_folder}/recommendations.txt', 'w') as file:
        file.write(recommendations)

    print(recommendations)


generate_next_steps()
