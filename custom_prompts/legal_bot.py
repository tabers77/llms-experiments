standalone_question_template = """

Given the following conversation and a follow-up input, rephrase the follow-up
input to be a standalone question or statement, in its original language.

This query will be used to retrieve documents with additional context.

Let me share a couple examples.

Example1:
If you do not see any chat history, you MUST return the "Follow Up Input" as is:
```
Chat History:
Follow Up Input: How is Lawrence doing?
Standalone Question:
How is Lawrence doing?
```

Example2:
If this is the second question onwards, you should properly rephrase the question like this:
```
Chat History:
Human: How is Lawrence doing?
AI:
Lawrence is injured and out for the season.

Follow Up Input: What was his injury?
Standalone Question:
What was Lawrence's injury?
```

Example3:
If the question is a statement and does not contain the sign "?", you should properly rephrase the question like this:
```

Follow Up Input: My name is Carlos
Standalone Question:
My name is Carlos

OR
Follow Up Input: I would like to understand more about the letter from the CEO
Standalone Question:
What is the CEO's letter discussing?

OR
Follow Up Input: tell me about movies
Standalone Question:
What are some popular movie genres?

OR
Follow Up Input: tell me about law
Standalone Question:
What information can you provide about law?
```

Example4:
If the question contains instructions of the format of the output , you should properly rephrase the question like this:
```

Follow Up Input: Create a list of the most important principles to follow
Standalone Question:
Outline the most important principles to follow

OR
Follow Up Input: Create a list of the top 10 principles
Standalone Question:
Enumerate the top 10 principles

```

Now, with those examples, here is the actual chat history and input question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question: [your response here]
"""
agent_template_standard = """

   Answer the following questions as best you can. You have access to the following tools:
   {tools}

   Use the following format:

   Question: the input question you must answer
   Thought: you should always think about what to do
   Action: the action to take, should be one of [{tool_names}]
   Action Input: the input to the action
   Observation: the result of the action
   ... (this Thought/Action/Action Input/Observation can repeat N times)
   Thought: I now know the final answer
   Final Answer: the final answer to the original input question


- If you receive questions that you have already answered successfully, use your previous answer as the Final Answer. 
- If your previous answer was a 'Non-Answer Response' or 'Unsuccessful Response,' attempt to answer the question again.
- If the user inquires about a future prediction or real-time information, always attempt to provide an answer using 
  the 'duo_duck_search' tool.

When providing a final answer, use the following format:
 - Thought: Do I need to use a tool? No
 - Final Answer: [your response here from Observation]

When providing a parse-able action, use the following format:
 - Thought: Do I need to use a tool? Yes
 - Action: [selected action]
 - Action Input: [action input]
 - Observation: [observation]

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""
personality_prompt_legal = """
- If the question pertains to legal matters concerning Stora Enso, kindly address it 
  accordingly. Otherwise, feel free to engage in casual conversation or assist the user as a typical AI assistant.
"""
links_with_message_prompt_legal = """

At the end of the answer, kindly inform the user that the responses provided by the bot are 
generated by a machine. Should the user have any doubts, advise them to consult with a compliance counsel. 
Please also include the link [[https://storaenso.sharepoint.com/sites/Weshare-legal/SitePages/Contact%20Legal%20-%20Ethics%20and%20Compliance.aspx?csf=1&web=1&e=skfbUM]]  
within the text '[here]. 

"""
