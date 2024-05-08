import logging
from _operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, format_document, ChatPromptTemplate

import custom_prompts.legal_bot
# from custom_prompts import prompts as p_templates

from typing import Any


# # ------------------------------------------------
#
# class AzureChatOpenAIWithTooling(AzureChatOpenAI):
#     """AzureChatOpenAI with a patch to support functions.
#
#     Function calling: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling
#
#     Currently only a single function call is supported.
#     If multiple function calls are returned by the model, only the first one is used.
#     """
#
#     def _generate(self, messages, stop=None, run_manager=None, stream=None, **kwargs):
#         if "functions" in kwargs:
#             kwargs["tools"] = [
#                 {"type": "function", "function": f} for f in kwargs.pop("functions")
#             ]
#         return super()._generate(messages, stop, run_manager, stream, **kwargs)
#
#     def _create_message_dicts(self, messages, stop):
#         dicts, params = super()._create_message_dicts(messages, stop)
#         latest_call_id = {}
#         for d in dicts:
#             if "function_call" in d:
#                 # Record the ID for future use
#                 latest_call_id[d["function_call"]["name"]] = d["function_call"]["id"]
#                 # Convert back to tool call
#                 d["tool_calls"] = [
#                     {
#                         "id": d["function_call"]["id"],
#                         "function": {
#                             k: v for k, v in d["function_call"].items() if k != "id"
#                         },
#                         "type": "function",
#                     }
#                 ]
#                 d.pop("function_call")
#
#             if d["role"] == "function":
#                 # Renaming as tool
#                 d["role"] = "tool"
#                 d["tool_call_id"] = latest_call_id[d["name"]]
#
#         return dicts, params
#
#     def _create_chat_result(self, response):
#         result = super()._create_chat_result(response)
#         for generation in result.generations:
#             if generation.message.additional_kwargs.get("tool_calls"):
#                 function_calls = [
#                     {**t["function"], "id": t["id"]}
#                     for t in generation.message.additional_kwargs.pop("tool_calls")
#                 ]
#                 # Only consider the first one.
#                 generation.message.additional_kwargs["function_call"] = function_calls[
#                     0
#                 ]
#         return result
#
# # --------------------------------------------------------

def get_llm_instance(configs):
    """
      Obtain the language model type based on the specified configuration.

      Args:
          configs : Configuration parameters.

      Returns:
          LLM: Language model based on the specified configuration.

      LLMs, or Large Language Models, are like supercharged text generators. They understand language,
      complete sentences, and can write various types of content. However, they are best suited for pure text-based
      tasks.

      If you need an AI that feels like a real conversation partner, then you need a Chat Model. Chat models
      understand context, remember past interactions, and are specifically designed for conversational flow. They are
      best suited for chatbots, virtual assistants, or any application requiring continuous dialogue.

      Usage:
          - For simple applications, using an LLM is suitable.
          - For applications requiring conversational flow, a Chat Model is recommended.
      """


    if configs.llm_type == 'azure_chat_openai':
        return AzureChatOpenAI(
            azure_deployment=configs.llm_deployment,
            openai_api_version=configs.openai_api_version,
            temperature=0
            # max_tokens=None  # configs.llm_max_tokens   # TEST
        )
    elif configs.llm_type == 'azure_openai':
        return AzureOpenAI(
            openai_api_type="azure_ad",
            deployment_name=configs.llm_deployment  # Name of the deployment for identification
        )

    elif configs.llm_type == 'hugging_face':
        return HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    else:
        raise ValueError('LLM type is not recognized')


## TODO: UPDATE THIS FUNCTION
# def initialize_chatbot_components(configs: conf_.ChatbotConfigs) -> tuple:
#     """
#     Initializes the chatbot components including vector store, retriever, and language model.
#
#     Args:
#         configs (object): Configuration parameters for the chatbot.
#
#     Returns:
#         tuple: A tuple containing the initialized vector store, retriever, and language model.
#     """
#     text_preprocessor = TextPreprocessor()
#
#     # TODO: FIX THIS
#     text_chunks = text_preprocessor.split_text_into_chunks_and_metas_wrap(return_chunks=True)
#
#     vectorstore, embeddings = emb_utils.get_vectorstore(configs, text_chunks, return_embeddings=True)
#
#     retriever = vectorstore.as_retriever()
#
#     llm = get_llm_type(configs)
#
#     logging.info('Initial chatbot components initialized')
#
#     return vectorstore, embeddings, retriever, llm


class RagChainWrapper:

    def __init__(self, retriever: Any, language_model: Any, loaded_memory: Any, answer_prompt: str) -> None:
        """
        Initializes a RagChainTool instance.

        Args:
            retriever: The retriever component responsible for retrieving relevant information.
            language_model: The language model component responsible for generating responses.
            loaded_memory: The loaded memory object containing conversation history or prior knowledge.
            answer_prompt: A prompt to guide the generation of answers for the language model.
        """

        self.retriever = retriever
        self.language_model = language_model
        self.loaded_memory = loaded_memory
        self.answer_prompt = answer_prompt

    def initialize_rag_with_memory(self) -> Any:
        """
        Initializes a RAG (Retriever-Augmented Generation) conversation chain with memory.

        Returns:
            Any: The initialized RAG conversation chain with memory.
        """

        # ----------------------------------------------
        # 1. Get chat history , standalone_question
        # - Convert your question to standalone_question using base_template
        # - Observe that standalone_question_template is a fixed parameter
        # ----------------------------------------------
        condense_question_prompt = PromptTemplate.from_template(custom_prompts.legal_bot.standalone_question_template)

        standalone_question = {
            "standalone_question": {
                                       "question": lambda x: x["question"],
                                       "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                                   }
                                   | condense_question_prompt
                                   | self.language_model
                                   | StrOutputParser(),
        }

        # ---------------------------------------------------
        # 2. Now we retrieve the documents from the pdfs
        # Here is the context being passed from the retriever
        # ---------------------------------------------------

        retrieved_documents = {
            "docs": itemgetter("standalone_question") | self.retriever,
            "question": lambda x: x["standalone_question"],
        }

        # --------------------------------------------------
        # 3. Now we construct the inputs for the final prompt
        # --------------------------------------------------
        default_document_prompt = PromptTemplate.from_template(template="{page_content}")

        def _combine_documents(
                docs,
                document_prompt=default_document_prompt,
                document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]

            return document_separator.join(doc_strings)

        final_inputs = {
            "context": lambda x: _combine_documents(docs=x["docs"]),
            "question": itemgetter("question"),
        }

        # -------------------------------------------------------
        # 4. And finally, we do the part that returns the answers
        # -------------------------------------------------------
        answer_prompt = ChatPromptTemplate.from_template(self.answer_prompt)

        answer = {
            "answer": final_inputs | answer_prompt | self.language_model,
            "docs": itemgetter("docs")}

        # Use the following line for debugging
        # standalone_chain = loaded_memory | standalone_question  # TEST

        final_chain = self.loaded_memory | standalone_question | retrieved_documents | answer

        return final_chain

    def run(self, user_question: str, run_for_agent: bool = True) -> Any:
        """
        Runs the RagChainTool to respond to the user's question.

        Args:
            user_question (str): The question asked by the user.
            run_for_agent (bool, optional): Whether to run the tool for the agent. Defaults to True.

        Returns:
            Any: The response to the user's question.
        """

        inputs = {"question": user_question}

        if run_for_agent:
            return self.initialize_rag_with_memory().invoke(inputs)["answer"].content

        return self.initialize_rag_with_memory().invoke(inputs)


# TODO: REFACTOR OR REMOVE
class BaseTool:
    """A class representing a tool for executing conversation chains."""

    def __init__(self, chain, chain_type='ConversationChain'):
        """
        Initialize the BaseTool with a given chain.

        :param chain: ConversationChain
            The conversation chain to use.
        """
        self.chain = chain
        self.chain_type = chain_type

    def run(self, user_question: str):
        """
          Execute the chain with the given user_question and return the response.

          :param user_question: str
              The user's question.
          :param chain_type: str, default='ConversationChain'
              The type of the chain to execute.
          :return: str
              The response from the chain.
          """

        # inputs = {"question": user_question}
        if self.chain_type == 'ConversationChain':
            return self.chain.invoke(user_question)["response"]
        else:
            raise ValueError('Chain type is not recognized')


class BaseChain(BaseTool):
    """A class representing a base conversation chain."""

    def __init__(self, chain, chain_type):
        """
        Initialize the BaseChain with a given chain.

        :param chain: ConversationChain
            The conversation chain to use.
        """
        super().__init__(chain, chain_type)
        self.chain = chain
        self.chain_type = chain_type


# --------
# MEMORIES
# --------
class CustomConversationBufferMemory(ConversationBufferMemory):

    def custom_clear(self, llm, max_token_limit=5000):
        """
        Clear the buffer based on the specified token limit.

        Args:
            llm: Language model used to calculate the number of tokens from messages.
            max_token_limit (int, optional): The maximum token limit allowed for the buffer. Defaults to 5000.
        """
        buffer = self.chat_memory.messages
        curr_buffer_length = llm.get_num_tokens_from_messages(buffer)
        while curr_buffer_length > max_token_limit:
            logging.info(f'curr_buffer_length has passed the {max_token_limit} max_token_limit')
            buffer.pop(0)
            curr_buffer_length = llm.get_num_tokens_from_messages(buffer)
            logging.info(f'curr_buffer_length: {curr_buffer_length}')
