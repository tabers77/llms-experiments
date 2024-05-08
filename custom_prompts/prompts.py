from custom_prompts.legal_bot import personality_prompt_legal, links_with_message_prompt_legal

# --------------------------------------------------------------------
# ************************* CODING  *************************
# --------------------------------------------------------------------


# TODO: TESTING THIS TEMPLATE

# USED FOR EXPERIMENTATION

# --------------------------------------------------------------------
# ************************* GENERAL  *************************
# --------------------------------------------------------------------
intro_prompt_general = """
- Please provide an answer based on the following context in its original language.
"""

formatting_prompt_general = """
- Present divided paragraphs in a visually appealing manner to enhance readability. Use appropriate spacing and 
  indentation to distinguish between paragraphs, facilitating comprehension.

- Please ensure that any crucial terms or concepts are bolded or italicized in the prompt text using Markdown syntax.
  For example: 
  Please explain the process of **photosynthesis** in plants.
  Please explain the process of **_photosynthesis_** in plants.

  In this example, the term "photosynthesis" is both bolded and italicized to make it stand out as a crucial term in the
  prompt

- If the question does not require a lengthy answer, please ensure that your responses are brief and to the point. 
"""

SUFFIX_GENERAL = """
Context: {context}
Question: {question}

Answer in Markdown:
"""

# --------------------------------------------------------------------
# ************************* PATENT  *************************
# --------------------------------------------------------------------

personality_prompt_patent = """
- If the question relates to internal patent claims and responses, kindly address it 
accordingly. Otherwise, feel free to engage in casual conversation or assist the user as a typical AI assistant.
"""


def create_answer_prompt(personality_prompt: str = None, links_with_message_prompt: str = None) -> str:
    """
    Creates a combined prompt for the GPT model.

    Args:
        personality_prompt (str, optional): The prompt related to the personality of the model. Defaults to None.
        links_with_message_prompt (str, optional): The prompt related to links with messages. Defaults to None.

    Returns:
        str: The final combined prompt for the GPT model.
    """
    personality_prompt = personality_prompt if personality_prompt is not None else ""
    links_with_message_prompt = links_with_message_prompt if links_with_message_prompt is not None else ""

    final_prompt = intro_prompt_general + personality_prompt + formatting_prompt_general + \
                   links_with_message_prompt + SUFFIX_GENERAL

    return final_prompt


def create_agent_base_prompt():
    # TODO: ADD FUNCTION TO AUTOMATE THE PROMPT GENERATION
    pass


class Prompts:
    legal_answer_prompt = create_answer_prompt(personality_prompt=personality_prompt_legal,
                                               links_with_message_prompt=links_with_message_prompt_legal
                                               )

    patents_answer_prompt = create_answer_prompt(personality_prompt=personality_prompt_patent,
                                                 )
