"""
src/debate_agent_lc.py

Defines a DebateAgentLC that:
  - uses ConversationBufferMemory to track conversation,
  - uses ChatPromptTemplate with a single user 'input',
  - merges 'stance_description' and 'agent_name' into the system context
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import LLMChain
from models import create_langchain_llm

class DebateAgentLC:
    """
    A single debate agent that:
      - has a stance (pro or con) described in a system message,
      - keeps a conversation buffer memory with user input key='input',
      - provides respond(input_text) to produce a new agent message.
    """

    def __init__(
        self,
        agent_name: str,          # e.g. "ProAgent" or "ConAgent"
        model_name: str,          # e.g. "o3-mini", "gemini-2.0-flash", etc.
        stance_description: str,  # e.g. "You are PRO on the topic..."
        temperature: float = 0.7
    ):
        self.agent_name = agent_name
        self.model_name = model_name
        self.stance_description = stance_description
        self.temperature = temperature

        # 1. Create the LLM using the official integration in your create_langchain_llm
        self.llm = create_langchain_llm(model_name, temperature=temperature)

        # 2. Conversation memory, expecting a single user input key named "input"
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",  # We'll reference this in the prompt as {chat_history}
            input_key="input",          # The user query is under "input"
            output_key="response"       # The LLM's response will be stored under "response"
        )

        # 3. We'll build a ChatPromptTemplate with system + memory + user
        #    The system message holds the stance context & agent identity.
        system_template = (
            "You are {agent_name}.\n"
            "{stance_description}\n"
            "Maintain a coherent multi-turn conversation and respond helpfully."
        )
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        # The conversation memory is inserted as a placeholder:
        memory_placeholder = MessagesPlaceholder(variable_name="chat_history")
        # The user's new message is the 'input':
        human_prompt = HumanMessagePromptTemplate.from_template("{input}")

        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            memory_placeholder,
            human_prompt
        ])

        # 4. Create an LLMChain from these pieces
        self.chain = LLMChain(
            llm=self.llm,
            prompt=chat_prompt,
            memory=self.memory,
            output_key="response"
        )

    def add_opponent_message(self, text: str):
        """
        Add the opponent's statement as a 'HumanMessage' in this agent's memory.
        This sets up the next turn so that when we call respond(...),
        the agent sees 'input' = user text.
        """
        # We directly append to chat_history for the memory. Alternatively, we can call chain's memory.save_context().
        self.memory.chat_memory.messages.append(HumanMessage(content=text))

    def add_self_reply(self, text: str):
        """
        Add the agent's own prior reply as an 'AIMessage' in this agent's memory.
        This ensures the memory has a complete record.
        """
        self.memory.chat_memory.messages.append(AIMessage(content=text))

    def respond(self, user_input: str = "") -> str:
        """
        Generate the agent's next response. We pass user_input as "input" to the chain.
        If user_input is empty, we rely on the previously appended 'HumanMessage' in memory.
        If non-empty, it's as if the agent sees a new user query in the same turn.
        :return: The agent's text response
        """
        # .invoke() returns a dict with {"response": "..."} if output_key="response"
        outputs = self.chain.invoke({"agent_name": self.agent_name,
                                     "stance_description": self.stance_description,
                                     "input": user_input})
        # The memory will store this user input as well as the LLM's new response automatically
        return outputs["response"].strip()

    def get_full_memory(self):
        """
        Return the entire conversation from this agent's perspective.
        """
        return self.memory.chat_memory.messages