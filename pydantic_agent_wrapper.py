import asyncio
from typing import Tuple
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.providers.openai import OpenAIProvider

class Pydantic_Agent:
    """
    A wrapper class for interacting with a Pydantic-AI language model agent.
    
    Handles prompt input, history tracking, and real-time streaming of model responses.
    """
    def __init__(self, model_name: str, url: str):
        """
        Initializes the agent with a specified model and base URL.

        Args:
            model_name: The name of the LLM model (e.g., `gemma3:4b`).
            url: The base URL for the model OpenAI endpoint.
        """
        self.model_name = model_name
        self.history: list[ModelRequest | ModelResponse] | None = None
        self.is_active: bool = True

        _model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=url)
        )
        self._agent = Agent(model=_model)

    def _get_user_prompt(self) -> str:
        """
        Prompts the user for input in the terminal.

        Returns:
            The user's text input as a string.
        """
        prompt = input(f"Make a request from {self.model_name}.\n> ")
        # NOTE: Only for CLI testing
        if prompt == "quit":
            exit()
        return prompt
    
    def compile_history(
        self,
        new_messages: list[ModelRequest | ModelResponse]
    ) -> list[ModelRequest | ModelResponse]:
        """
        Combines the existing history with newly received messages.

        Args:
            new_messages: The list of ModelRequest or ModelResponse objects to add.

        Returns:
            The updated conversation history.
        """
        if self.history:
            return self.history + new_messages
        else:
            return new_messages 

    async def _handle_llm_query(
            self,
            prompt: str,
            history: list[ModelRequest | ModelResponse] | None
    ) -> Tuple[list[ModelRequest | ModelResponse], bool]:
        """
        Streams a response from the LLM using the given prompt and optional history.

        Args:
            prompt: The user's prompt string.
            history: A list of previous model messages to provide context.

        Returns:
            A tuple containing the list of new messages and a boolean indicating if the model completed its response.
        """
        print(f"{model_name}:\n> ", end="", flush=True)
        async with self._agent.run_stream(prompt, message_history=history) as response:
            string_index = 0
            async for data in response.stream_text(debounce_by=0.1):
                # Output text stream by printing at the end of previous message
                print(data[string_index:], end="", flush=True)
                string_index = len(data)
        print() # New line

        return ([message for message in response.all_messages()], response.is_complete)
    
    def run_query(self) -> list[ModelRequest | ModelResponse]:
        """
        Runs the full user input and streaming response loop once.

        Returns:
            The new list of model messages resulting from the user query.
        """
        messages, is_complete = asyncio.run(self._handle_llm_query(
            prompt=self._get_user_prompt(),
            history=self.history
        ))
        self.history = self.compile_history(new_messages=messages)
        self.is_active != is_complete
        return messages
    

if __name__ == "__main__":
    url = "http://localhost:11434/v1"
    model_name = "gemma3:1b"

    gemma3_1b = Pydantic_Agent(model_name=model_name, url=url)

    while True: # Not an infinate loop. See Pydantic_Agent._get_user_prompt()
        print("Type `quit` to end program.")
        _ = gemma3_1b.run_query()
