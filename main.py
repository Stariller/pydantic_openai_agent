import asyncio
from typing import Tuple

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.providers.openai import OpenAIProvider

from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg

class RAGInput(BaseModel):
    query: str  # Explicit parameter the LLM sees and can fill

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

    # @self._agent.tool
    def rag_tool(self, prompt: str) -> str:
            """
            """
            # print("RAG Tool is running with prompt:", prompt)
            # Embbed user prompt
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeded_prompt = model.encode(prompt).tolist()

            # Ask pg server to run cosine similarity
            with psycopg.connect("dbname=ragdb user=michael host=localhost port=5432") as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM documents
                    ORDER BY embedding <#> %s::vector
                    LIMIT 5;
                    """,
                    (('[' + ','.join(map(str, embeded_prompt)) + ']',))
                )
                results = cur.fetchall()

            if not results:
                return "No relevant context found during RAG."

            # print("\n\n".join(f"[Page {meta.get('page', '?')}] {content}"for content, meta in results))

            return "\n\n".join(
                f"[Page {meta.get('page', '?')}] {content}"
                for content, meta in results
            )

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
    
    def _compile_history(
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

    async def _handle_llm_query(self, prompt: str) -> Tuple[list[ModelRequest | ModelResponse], bool]:
        """
        Streams a response from the LLM using the given prompt and optional from self.history.

        Args:
            prompt: The user's prompt string.

        Returns:
            A tuple containing the list of new messages and a boolean indicating if the model completed its response.
        """
        print(f"{self.model_name}:\n> ", end="", flush=True)
        async with self._agent.run_stream(prompt, message_history=self.history) as response:
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
        user_prompt = self._get_user_prompt()
        rag_results = self.rag_tool(prompt=user_prompt)
        # NOTE: I have been trying so many ways to get this to work.
        formatted_message = f"""
You are an assistant with access to a retrieval tool (`rag_tool`). 
Use this tool first to retrieve relevant context from documents.

After retrieving the context, explicitly incorporate it into your response.

Here are the RAG results: {rag_results}

User Prompt:
{user_prompt}
        """
        messages, is_complete = asyncio.run(self._handle_llm_query(prompt=formatted_message))
        self.history = self._compile_history(new_messages=messages)
        self.is_active != is_complete
        return messages
    

if __name__ == "__main__":
    url = "http://localhost:11434/v1"
    
    gemma3 = Pydantic_Agent(model_name="gemma3:12b", url=url)

    while True: # Not an infinate loop. See Pydantic_Agent._get_user_prompt()
        print("Type `quit` to end program.")
        _ = gemma3.run_query()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine similarity function.
    cos(θ) = (A ⋅ B) / (||A|| * ||B||)
    Where:
        θ is our angle between the two vectors.
        A & B are vectors of any size.
        (A ⋅ B) are the dot product of the two vectors.
        ||A|| & ||B|| are magnitudes of the vectors.
        Magnitude is the root of the sum of the squares. ||A|| = sqrt(1^2 + 2^2 + ... n^2)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # NOTE: I guess postgreSQL supports the cosine similarity search so I won't be using this.
    #       but I want to leave this here so I know what math it's doing.
