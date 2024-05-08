from dataclasses import dataclass
from dataclasses_json import dataclass_json
from langchain_text_splitters import CharacterTextSplitter

from dotenv import load_dotenv
import os


@dataclass_json
@dataclass(frozen=True)
class ChatbotConfigs:
    load_dotenv()  # Load environment variables from .env file
    embeddings_deployment: str = "text-embedding-ada-002"
    llm_deployment: str = "langchain_model"
    llm_type: str = 'azure_openai'
    openai_api_version: str = "2023-07-01-preview"

    storage_type: str = 'blob'
    blob_container_name: str = "policies-container"
    blob_connection_string = os.getenv("CONNECTION_STRING")
    pg_connection_string: str = f"postgresql://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    pg_collection_name: str = "state_of_union_vectors2"

    is_authentication_for_prod: bool = True
    saver_version: bool = False

    answer_prompt: str = None
    text_splitter: CharacterTextSplitter = CharacterTextSplitter
    embeddings_type: str = 'azure_openai'

# @dataclass_json


# @dataclass(frozen=True)
class LLMConfigs:
    load_dotenv()  # Load environment variables from .env file
    embeddings_deployment: str = "text-embedding-ada-002"
    llm_deployment: str = "langchain_model"
    llm_type: str = 'azure_chat_openai'
    embeddings_type: str = 'azure_openai'
    openai_api_version: str = "2023-07-01-preview"


# @dataclass_json
# @dataclass(frozen=True)
class DatabaseConfigs:
    storage_type: str = 'blob'
    blob_container_name: str = "policies-container"
    blob_connection_string = os.getenv("CONNECTION_STRING")
    pg_connection_string: str = f"postgresql://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    pg_collection_name: str = None


# @dataclass_json
# @dataclass(frozen=True)
class FlaskAppConfigs:
    is_authentication_for_prod: bool = True
    saver_version: bool = False


# @dataclass_json
# @dataclass(frozen=True)
# TODO: ADD PROMPTS WRAPPER FROM FILE prompts.py
class PromptConfigs:
    answer_prompt: str = None
    agent_template_standard: str = None


@dataclass_json
@dataclass(frozen=True)
class Cfg:
    llm_configs: LLMConfigs = LLMConfigs()
    database_configs: DatabaseConfigs = DatabaseConfigs()
    flask_app_configs: FlaskAppConfigs = FlaskAppConfigs()
    prompt_configs: PromptConfigs = PromptConfigs()
