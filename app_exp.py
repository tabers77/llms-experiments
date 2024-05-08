""" NOTE: THIS A TEMPORARY PY FILE FOR EXPERIMENTATION--> """
import logging
from pathlib import Path
import pandas as pd

from flask import Flask, render_template, request, current_app
from flask_session import Session
from dotenv import load_dotenv
import custom_prompts
import custom_prompts.coder_bot
from conf import configs, app_config
import lang_models as comps

from ms_identity_web import IdentityWebPython
from ms_identity_web.adapters import FlaskContextAdapter
from ms_identity_web.configuration import AADConfig

# from langchain.chains import ConversationChain
from experimental import agent_tools
from utils import extract_json_data

logging.basicConfig(level=logging.INFO)

# -----------------------------------------------


import re


def check_for_keywords(text):
    keywords = re.compile(
        r'\b(?:plot|plots|visualization|visualizations|graph|graphs|barplot|distplot|related|dis?plot)\b',
        re.IGNORECASE)
    return bool(re.search(keywords, text))


# -----------------------------------------------
class FlaskWrapper:
    """
    This class serves as a wrapper for initializing a Flask application with specific configurations for a chatbot
    system. The main functionality of this class includes:

    1. Initialization: The constructor initializes the FlaskWrapper object with provided configurations.

    2. initialize_steps method:
   - Loads environment variables from a .env file.
   - Initializes a Flask application.
   - Loads Flask configuration from an external file.
   - Sets up server-side session management for the application.
   - Registers error handlers for exceptions.
   - Parses Azure Active Directory (AAD) configurations from a JSON file.
   - Configures logging levels for the Flask application.
   - Configures middleware for handling authentication in production environments.
   - Instantiates an adapter for FlaskContextAdapter and IdentityWebPython for Microsoft Identity Web integration.
   - Determines the appropriate conversation chain based on configurations.
   - Defines functions for obtaining responses from the chatbot, including handling responses with or without RAG
    (Retrieval-Augmented Generation).
   - Defines routes for handling HTTP requests, including getting bot responses, rendering index pages, and displaying
     token details.
   - Returns the initialized Flask application.
    """

    def __init__(self, flask_wrapper_configs):
        self.memory_chit_chat = None
        self.loaded_memory_rag = None
        self.memory_rag = None
        self.flask_wrapper_configs = flask_wrapper_configs
        self.pandas_df = None
        self.pandas_agent = None
        self.user_question = None  # TEST

    def initialize_steps(self) -> Flask:

        load_dotenv()
        # -----------------------
        # 1. Initialize Flask app
        # -----------------------
        app = Flask(__name__, root_path=str(Path(__file__).parent))
        app.config.from_object(app_config)  # load Flask configuration file (e.g., session configs)
        Session(app)  # init the serverside session for the app: this is required due to large cookie size
        # -----------------------
        # 2. Azure Authentication
        # -----------------------
        aad_configuration = AADConfig.parse_json('conf/aad.config.json')  # parse the aad configs
        app.logger.level = logging.INFO  # can set to DEBUG for verbose logs

        if self.flask_wrapper_configs.is_authentication_for_prod:
            # print('running production')
            # The following is required to run on Azure App Service or any other host with reverse proxy:
            from werkzeug.middleware.proxy_fix import ProxyFix
            app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
            # print('app.wsgi_app ', app.wsgi_app)
            # Use client credential from outside the config file, if available.

        AADConfig.sanity_check_configs(aad_configuration)
        adapter = FlaskContextAdapter(app)  # ms identity web for python: instantiate the flask adapter
        ms_identity_web = IdentityWebPython(aad_configuration, adapter)

        if not self.flask_wrapper_configs.saver_version:
            # ----------------------------
            # 3. Initialize Memory Objects
            # ----------------------------
            # self.memory_rag = ConversationBufferMemory(return_messages=True,
            #                                            output_key="answer",
            #                                            input_key="question"
            #                                            )
            # self.loaded_memory_rag = RunnablePassthrough.assign(
            #     chat_history=RunnableLambda(self.memory_rag.load_memory_variables) | itemgetter("history"),
            # )
            # self.memory_chit_chat = ConversationBufferMemory()
            # self.memory_chit_chat = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            llm = comps.get_llm_instance(configs=self.flask_wrapper_configs)

            self.memory_chit_chat = comps.CustomConversationBufferMemory(llm_custom=llm,
                                                                         memory_key="chat_history",
                                                                         return_messages=True
                                                                         )

            # self.memory = ConversationTokenBufferMemory(llm=llm,
            #                                             max_token_limit=3,
            #                                             output_key="answer",
            #                                             input_key="question")
            #

            # ------------------------------------------
            # 4. Initialize necessary chatbot components -  Initialize chains
            # ------------------------------------------

            # # Note: Uncomment this line for testing:
            # vectorstore, embeddings, retriever, llm = chatbot_components.initialize_chatbot_components(
            #     configs=self.flask_wrapper_configs)
            # # --------------
            # # 4.1 RAG CHAIN
            # # --------------
            # retriever = chatbot_components.initialize_pg_vector(
            #     configs=self.flask_wrapper_configs)

            # We initialize the chains/tools when the application is started to avoid running the function every time we have
            # a get response

            # rag_chain_tool = chatbot_components.RagChainWrapper(retriever, llm, self.loaded_memory_rag)

            # --------------
            # 4.2 CHITCHAT
            # --------------

            # from langchain.prompts.prompt import PromptTemplate
            # answer_prompt = PromptTemplate(input_variables=["history", "input"],
            #                                template=prompt_templates.code_assistant_bot_template)
            #
            # code_assistant = ConversationChain(
            #     llm=llm,
            #     prompt=answer_prompt
            # )
            #
            # code_assistant_tool = agent_tools.BaseChain(code_assistant)

            from langchain_community.tools import DuckDuckGoSearchRun

            debug_researcher = agent_tools.get_custom_tool(func=DuckDuckGoSearchRun(),
                                                           name="debug_researcher",
                                                           description="useful for searching the internet for solution to code bugs"

                                                           )

            tools = agent_tools.initialize_tools(
                tools_names_to_use=['programmer', 'python_repl'],  # 'duo_duck_search', ' 'python_repl'
                language_model=llm
            )

            agent_executor = agent_tools.initialize_agent_executor(language_model=llm,
                                                                   tools=tools,
                                                                   memory=self.memory_chit_chat,
                                                                   custom_template=custom_prompts.coder_bot.agent_template_code_genie_test2)

            # ---------------------------------------
            # EXPERIMENTAL
            tool_plot = agent_tools.load_llms_chain_as_tools(llm,
                                                             template=custom_prompts.coder_bot.agent_template_plot_displayer,
                                                             )

            # --------------------------------------------------------

            def get_response(user_question: str) -> str:
                raise NotImplementedError
                # TODO: Implement this code
                # conversation_history = "Previous conversation history here."
                # # Query the conversation chain to get a response
                # response = chain_from_vectorstore({'chat_history': conversation_history, 'question': user_question})
                #
                # # Process the response
                # response_text = response['answer']
                # return response_text

            def get_response_rag(user_question):  # chain_from_vectorstore
                raise NotImplementedError
                # # Query the conversation chain to get a response
                # rag_chain = chain_from_vectorstore
                # # inputs = {"question": str(user_question)} # TEST
                # # response_text = rag_chain.invoke(inputs) # test
                # response_text = rag_chain.invoke(user_question)
                # # Process the response
                # return response_text

            def get_chatbot_response(user_question, chat_mode):

                inputs = {"input": user_question}

                # # TODO: MOVE TO CHATBOT COMPONENTS

                # if check_for_keywords(user_question):
                if self.pandas_df is not None and chat_mode == 'EDA':
                    self.pandas_agent = agent_tools.create_pandas_dataframe_agent_custom(llm=llm,
                                                                                         df=self.pandas_df,
                                                                                         memory=self.memory_chit_chat,
                                                                                         verbose=True,
                                                                                         # return_intermediate_steps=True # this makes no diference
                                                                                         )

                    return self.pandas_agent.invoke(user_question)['output']

                result = agent_executor.invoke(inputs)['output']

                self.memory_chit_chat.save_context(inputs, {"answer": result})

                self.memory_chit_chat.custom_clear(llm=llm)

                return result

                # -------------------------------

                # if chat_mode == 'LegalBotExperimental':
                #
                #     inputs = {"input": user_question}
                #
                #     result = agent_executor.invoke(inputs)['output']
                #
                #     self.memory_chit_chat.save_context(inputs, {"answer": result})
                #
                #     return result
                #
                # elif chat_mode == 'LegalBot':
                #
                #     result = rag_chain_tool.run(user_question, run_for_agent=False)
                #
                #     sources = generate_sources(result)
                #     # --------------------------------------
                #     # # Use only for debugging !!!
                #     # result_test = test_chain.invoke(inputs)  # TEST
                #     # print('STANDALONE', result_test)
                #     # --------------------------------------
                #     self.memory_rag.save_context(inputs, {"answer": result["answer"].content})
                #     self.memory_rag.load_memory_variables({})
                #
                #     return result["answer"].content + sources
                #
                # else:
                #     raise ValueError('Chat Mode not recognized..')

            @app.route("/get")
            # Function for the bot response
            def get_chatbot_response_endpoint():  # -> str:
                from flask import jsonify

                # chat_mode = request.args.get('option')
                # user_question = request.args.get('msg')
                #
                # response_text = get_chatbot_response(user_question=user_question,
                #                                      chat_mode=chat_mode
                #                                      )

                # return response_text

                # # --------------------------------------------------------------------------
                # # TEST MODE
                # response_text = 'parameter1 and parameter2 are the parameters that you want to use to generate {"x": [1, 2, 3, 4, 5],"y": [10, 20, 30, 40, 50],"type": "line","orientation": "vertical"}'
                #
                # remaining_text, json_data = extract_json_data(response_text)
                #
                # print('remaining_text', remaining_text)
                # print('json_data', json_data)

                chat_mode = request.args.get('option')
                self.user_question = request.args.get('msg')

                self.response_text = get_chatbot_response(user_question=self.user_question,
                                                          chat_mode=chat_mode
                                                          )

                return self.response_text

                # #  --------------------------------------------------------------------------

        @app.route('/get_graph')
        def get_graph():
            from flask import jsonify
            print('check_for_keywords(self.user_question)', check_for_keywords(self.user_question))

            # if not check_for_keywords(self.user_question):
            #   return "", ""
            # else:
            # print('self.response_text', self.response_text)

            # --------------------------------------------
            # Experimental
            #   print('self.user_question', self.user_question)

            #  response_text = tool_plot.run(self.user_question)

            # print('response_text', response_text)

            # -------------------------------------------

            remaining_text, json_data = extract_json_data(self.response_text)

            # TEMP LAYOUT
            layout = {
                'title': 'Always Display the Modebar',
                'showlegend': False
            }

            json_data = {
                'data': json_data,
                'layout': layout
            }

            return jsonify([remaining_text, json_data])  # Return data as a list (or tuple) to be serialized as JSON

        # @app.route('/')
        # def index():
        #     return render_template('index_exp.html')

        @app.route('/', methods=['GET', 'POST'])
        @app.route('/sign_in_status')
        def index():
            if request.method == 'POST':
                # file = request.files['csvfile']
                csv_file = request.files['csvfile']
                self.pandas_df = pd.read_csv(csv_file)
                logging.info(f'csv file {csv_file} upload and pandas df created')

            return render_template('welcome_page_exp.html')

        @app.route('/token_details')
        @ms_identity_web.login_required  # <-- developer only needs to hook up login-required endpoint like this
        def token_details():
            print('token_details triggered')
            current_app.logger.info("token_details: user is authenticated, will display token details")
            return render_template('auth/token.html')

        return app

    # ---------------------------------------------
    # @flask_bot.route("/audio", methods=["POST"])
    # def handle_audio_input():
    #     audio_text = ut.transcribe_speech()
    #     if audio_text:
    #         response_text = get_response_rag(audio_text) if self.configs.use_rag else get_response(audio_text)
    #         return response_text
    #     else:
    #         return "Sorry, I could not understand the audio."


configs_chat = configs.ChatbotConfigs(embeddings_deployment="text-embedding-ada-002",
                                      llm_deployment="langchain_model",
                                      llm_type='azure_chat_openai',  # 'hugging_face'  # 'azure_openai'
                                      is_authentication_for_prod=True,
                                      saver_version=False,
                                      pg_collection_name='emb_metadata_docs'  # 'test_emb_metadata'
                                      )
# from conf.configs import Cfg
#
# cfg_instance = Cfg()
# cfg_instance.llm_configs.llm_type = 'azure_chat_openai'
# cfg_instance.llm_configs.embeddings_type = 'azure_openai'
# cfg_instance.flask_app_configs.is_authentication_for_prod = True
# cfg_instance.flask_app_configs.saver_version = False
# cfg_instance.database_configs.pg_collection_name = 'emb_metadata_docs2'
# cfg_instance.prompt_configs.answer_prompt = custom_prompts.legal_bot.agent_template_legal

flask_wrapper = FlaskWrapper(flask_wrapper_configs=configs_chat)

if __name__ == "__main__":
    flask_bot = flask_wrapper.initialize_steps()
    flask_bot.run(host='0.0.0.0', port=80)  # debug=True
