'''
src/agent.py

DebateAgentLC: A debate agent that uses LangChain's message history for conversation tracking.
'''
from typing import List, Optional, Dict, Any
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.models import create_langchain_llm

class DebateAgentLC:
    """
    A debate agent that:
      - uses LangChain's message history for conversation tracking,
      - compiles a chain via the | operator (creating a RunnableSequence),
      - wraps the chain with RunnableWithMessageHistory for message persistence.
    """
    def __init__(
        self,
        agent_name: str,          # e.g. "ProAgent" or "ConAgent"
        model_name: str,          # e.g. "o3-mini", "gemini-2.0-flash", etc.
        stance_description: str,  # e.g. "You are PRO on the topic..."
        temperature: float = 0.7,
        verbose: bool = False     # Set to True for debug prints
    ):
        self.agent_name = agent_name
        self.model_name = model_name
        self.stance_description = stance_description
        self.temperature = temperature
        self.verbose = verbose
        
        # Create a unique ID for this agent instance
        self.instance_id = str(uuid.uuid4())[:8]
        self.session_id = f"{self.agent_name}_{self.instance_id}"

        # Create the LLM using your helper function
        self.llm = create_langchain_llm(model_name, temperature=temperature)

        # Create a message history store - this replaces ConversationBufferMemory
        self.message_history = ChatMessageHistory()
        
        # Store message histories in a dict so they can be accessed by session_id
        self.message_histories = {self.session_id: self.message_history}

        # Build the chat prompt template
        system_template = (
            "You are {agent_name}.\n"
            "{stance_description}\n"
            "Maintain a coherent multi-turn conversation and respond helpfully."
        )
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        memory_placeholder = MessagesPlaceholder(variable_name="chat_history")
        human_prompt = HumanMessagePromptTemplate.from_template("{input}")

        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            memory_placeholder,
            human_prompt
        ])

        # Compose the chain using the pipe operator
        base_chain = self.chat_prompt | self.llm
        
        # Define a message history getter function
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.message_histories:
                self.message_histories[session_id] = ChatMessageHistory()
            return self.message_histories[session_id]
        
        # Create the chain with message history
        self.chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output"
        )

    def _debug_print_prompt(self, variables: Dict[str, Any]):
        """
        Print the formatted prompt for debugging purposes.
        """
        if not self.verbose:
            return
            
        print(f"\n{'='*80}\n[DEBUG] {self.agent_name} PROMPT:")
        
        # Format and print the system message
        system_template = (
            "You are {agent_name}.\n"
            "{stance_description}\n"
            "Maintain a coherent multi-turn conversation and respond helpfully."
        )
        formatted_system = system_template.format(
            agent_name=variables["agent_name"],
            stance_description=variables["stance_description"]
        )
        print(f"SYSTEM:\n{formatted_system}\n")
        
        # Print the conversation history
        print("CONVERSATION HISTORY:")
        for msg in self.message_history.messages:
            prefix = "HUMAN" if isinstance(msg, HumanMessage) else "AI"
            print(f"{prefix}: {msg.content}")
        
        # Print the current input
        print(f"\nCURRENT INPUT: {variables['input']}")
        print(f"{'='*80}\n")

    def respond(self, user_input: str = "") -> str:
        """
        Generate the agent's next response.
        This method passes the agent's identity, stance, and the new user input.
        """
        # Create input variables
        variables = {
            "agent_name": self.agent_name,
            "stance_description": self.stance_description,
            "input": user_input,
        }
        
        # Debug print the full prompt
        self._debug_print_prompt(variables)
        
        # Invoke the chain with session_id to track conversation history
        outputs = self.chain.invoke(
            variables,
            config={"configurable": {"session_id": self.session_id}}
        )
        
        # Extract and return the AI's response
        response = outputs.content.strip() if hasattr(outputs, "content") else str(outputs)
        
        if self.verbose:
            print(f"\n[DEBUG] {self.agent_name} RESPONSE: {response}\n")
            
        return response

    def get_full_memory(self):
        """
        Return the full conversation history.
        """
        return self.message_history.messages