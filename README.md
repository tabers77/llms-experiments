# LLM Experiments
Our project aims to create intelligent question-answering bots using open-source technologies like Langchain. We're exploring various techniques, including RAG chains, memory management, object parsing, and embedding generation. Additionally, we're investigating the storage of embeddings in PostgreSQL databases and exploring advanced concepts such as agents, multi-agents, custom tools, and sequential chains.

Our focus extends to testing experimental methods and releases from Langchain, with the overarching goal of integrating reasoning into Large Language Models (LLMs).

Moreover, the project is tailored for Azure Cloud, leveraging Large Language Models like Azure ChatOpenAI and AzureOpenAI. We've developed the application in Python using the Flask framework, while the question-and-answer chatbot itself is implemented using Langchain. The frontend is built with CSS, JS, and HTML.

For deployment, we've containerized the application using Docker containers and deployed it onto Azure Web Apps.


## Table of Contents
- [Usage](#usage)

## Usage

  ```python
import logging

from flask_app import FlaskWrapper
from custom_prompts import prompts
from custom_prompts import legal_bot as legal_prompts
from conf.configs import Cfg

logging.basicConfig(level=logging.INFO)

cfg_instance = Cfg()
cfg_instance.llm_configs.llm_type = 'azure_chat_openai'
cfg_instance.llm_configs.embeddings_type = 'azure_openai'
cfg_instance.flask_app_configs.is_authentication_for_prod = True
cfg_instance.flask_app_configs.saver_version = False

cfg_instance.database_configs.pg_collection_name = 'emb_metadata_docs_patent'  # 'emb_metadata_docs2'
cfg_instance.prompt_configs.answer_prompt = prompts.Prompts.patents_answer_prompt  # prompts.Prompts.legal_answer_prompt
cfg_instance.prompt_configs.agent_template_legal = legal_prompts.agent_template_standard

flask_wrapper = FlaskWrapper(configs_=cfg_instance)

if __name__ == "__main__":
    flask_bot = flask_wrapper.initialize_steps()
    flask_bot.run(host='0.0.0.0', port=80)  # debug=True

   ```
