import streamlit as st

from langchain.agents import initialize_agent, AgentType, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

template = """You are customer assistant in Allegro marketplace. Your task is to understand what product the customer is looking for and advice when customer have problems.
You must define the most important parameters of the product that customer looks for. If customer does not provide them you should ask additional questions regarding the most important parameters to understand customer needs better - in this situation the set of questions to customer will be your final answer.
Advice to customer is also treated as final answer. 
Once you have a clear understanding, you should provide a product name with defined parameters. REMEMBER - the product should be generalized.
Your response should always be in Polish.
To check what parameters are important for given product or how to advice to customer you can use following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do or what question you want to ask to Customer
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question or questions about product parameters

Begin! Remember to answer in polish when giving your final answer.

Question: {input}
{agent_scratchpad}"""


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)



class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # Handle case when output doesn't match expected format
            return AgentFinish(
                return_values={"output": llm_output},
                log="Could not parse LLM output, returning as is.",
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    reset_conversation = st.button("Reset Conversation")  # Button to reset the conversation

# If the reset button is clicked, reset the messages in the session state
if reset_conversation:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Cześć, jestem Twoim asystentem w poszukiwaniach na Allegro. W czym mogę Ci pomóc?"}
    ]

if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
search = DuckDuckGoSearchRun(name="Search")

tools = [
    Tool(
        name = "Search",
        func=DuckDuckGoSearchRun(name="Search"),
        description="useful for when you need to find parameters of the product customer is looking for"
    )
]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)






st.title("Allegro Assistant")

"""
Place for transparent assumptions
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Cześć, jestem Twoim asystentem w poszukiwaniach na Allegro. W czym mogę Ci pomóc?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Szukam rowerka dla dziecka. Na co powinienem zwrócić uwagę?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent_executor.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


