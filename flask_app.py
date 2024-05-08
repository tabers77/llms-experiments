import logging
from _operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, render_template, current_app
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from ms_identity_web import IdentityWebPython
from ms_identity_web.adapters import FlaskContextAdapter
from ms_identity_web.configuration import AADConfig

import lang_models as comps
import readers_preprocessors
from experimental import agent_tools
from flask_session import Session

from conf import app_config
from conf.configs import Cfg
from utils import generate_sources


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

    def __init__(self, configs_):
        self.memory_chit_chat = None
        self.loaded_memory_rag = None
        self.memory_rag = None
        self.configs: Cfg = configs_

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

        if self.configs.flask_app_configs.is_authentication_for_prod:
            # print('running production')
            # The following is required to run on Azure App Service or any other host with reverse proxy:
            from werkzeug.middleware.proxy_fix import ProxyFix
            app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
            # print('app.wsgi_app ', app.wsgi_app)
            # Use client credential from outside the config file, if available.

        AADConfig.sanity_check_configs(aad_configuration)
        adapter = FlaskContextAdapter(app)  # ms identity web for python: instantiate the flask adapter
        ms_identity_web = IdentityWebPython(aad_configuration, adapter)

        if not self.configs.flask_app_configs.saver_version:

            # ----------------------------
            # 3. Initialize Memory Objects & llm
            # ----------------------------

            llm = comps.get_llm_instance(configs=self.configs.llm_configs)

            self.memory_rag = comps.CustomConversationBufferMemory(llm_custom=llm,
                                                                   return_messages=True,
                                                                   output_key="answer",
                                                                   input_key="question"
                                                                   )

            self.loaded_memory_rag = RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory_rag.load_memory_variables) | itemgetter("history"),
            )

            self.memory_chit_chat = comps.CustomConversationBufferMemory(llm_custom=llm,
                                                                         memory_key="chat_history",
                                                                         return_messages=True
                                                                         )

            # ------------------------------------------
            # 4. Initialize necessary chatbot components -  Initialize chains
            # ------------------------------------------

            # # Note: Uncomment this line for testing:
            # vectorstore, embeddings, retriever, llm = chatbot_components.initialize_chatbot_components(
            #     configs=self.flask_wrapper_configs)
            # --------------
            # 4.1 RAG CHAIN
            # --------------
            retriever = readers_preprocessors.initialize_pg_vector(configs=self.configs)

            # We initialize the chains/tools when the application is started to avoid running the function every time
            # we have a get response

            rag_chain_tool = comps.RagChainWrapper(retriever=retriever,
                                                   language_model=llm,
                                                   loaded_memory=self.loaded_memory_rag,
                                                   answer_prompt=self.configs.prompt_configs.answer_prompt
                                                   )

            # --------------
            # 4.2 CHITCHAT
            # --------------

            tools = agent_tools.initialize_tools(tools_names_to_use=['duo_duck_search', 'se_language_model'],
                                                 language_model=llm,
                                                 rag_chain_tool=rag_chain_tool
                                                 )

            agent_executor = agent_tools.initialize_agent_executor(language_model=llm,
                                                                   tools=tools,
                                                                   memory=self.memory_chit_chat,
                                                                   custom_template=self.configs.prompt_configs.agent_template_legal)

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

                inputs = {"question": user_question}

                if chat_mode == 'LegalBotExperimental':

                    inputs = {"input": user_question}

                    result = agent_executor.invoke(inputs)['output']

                    self.memory_chit_chat.save_context(inputs, {"answer": result})

                    # Clear the Chat Memory based on a token limit
                    self.memory_chit_chat.custom_clear(llm=llm)

                    return result

                elif chat_mode == 'LegalBot':

                    result = rag_chain_tool.run(user_question, run_for_agent=False)
                    # result = rag_chain_tool.run(user_question)  # TEST

                    sources = generate_sources(result)
                    # --------------------------------------
                    # TEST
                    # # Use only for debugging !!!
                    # result_test = test_chain.invoke(inputs)
                    # print('STANDALONE', result_test)
                    # --------------------------------------
                    self.memory_rag.save_context(inputs, {"answer": result["answer"].content})
                    self.memory_rag.load_memory_variables({})

                    # Clear the Chat Memory based on a token limit
                    self.memory_rag.custom_clear(llm=llm)

                    return result["answer"].content + sources

                else:
                    raise ValueError('Chat Mode not recognized..')

            @app.route("/get")
            # Function for the bot response
            def get_chatbot_response_endpoint() -> str:
                chat_mode = request.args.get('option')
                user_question = request.args.get('msg')

                response_text = get_chatbot_response(user_question=user_question,
                                                     chat_mode=chat_mode)

                return response_text

        @app.route('/')
        @app.route('/sign_in_status')
        def index():
            return render_template('welcome_page.html')

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
