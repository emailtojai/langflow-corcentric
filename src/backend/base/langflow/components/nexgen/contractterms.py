from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message
from langflow.schema import Data
from langchain_community.vectorstores import Qdrant
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import ChatPromptTemplate
from typing import List
import os
from dotenv import dotenv_values

# Load environment variables
dotenv_values()

class ContractPaymentTermsTool(Component):
    display_name = "Contract Payment Terms Tool"
    description = "A tool for agents to extract payment terms from contracts using Qdrant and Amazon Bedrock."
    documentation = "https://docs.langflow.org/components-custom-components"
    icon = "tool"
    name = "ContractPaymentTermsTool"

    inputs = [
        MessageTextInput(
            name="query",
            display_name="Query",
            info="The query to search for payment terms in contracts (e.g., 'What are the payment terms?').",
            required=True,
            tool_mode=True,  # Enable Tool Mode toggle
        ),
    ]

    outputs = [
        Output(display_name="Payment Terms", name="output", method="process_query"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use Bedrock embeddings instead of OpenAI
        self.embedding = BedrockEmbeddings(
            region_name=os.getenv("aws_region", "us-east-1"),  # Default to us-east-1 if not set
            model_id="amazon.titan-embed-text-v1"
        )
        # Use Bedrock LLM instead of OpenAI
        self.llm = ChatBedrock(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name=os.getenv("aws_region", "us-east-1"),
            streaming=True,
            verbose=False,
            model_kwargs={"max_tokens": 1000, "top_p": 0.9, "temperature": 0}
        )
        self.qdrant_url = "http://localhost:6333"  # Replace with your Qdrant URL
        self.collection_name = "procurement_contracts"

    def build_qdrant(self) -> Qdrant:
        """Initialize Qdrant vector store."""
        from qdrant_client import QdrantClient
        client = QdrantClient(url=self.qdrant_url)
        return Qdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embedding,
        )

    def format_documents(self, documents: List[Data]) -> str:
        """Format Qdrant search results into a string."""
        if not documents:
            return "No search results found."
        doc_texts = [doc.data.get("text", "No content") for doc in documents if isinstance(doc, Data)]
        return "\n".join([f"- {text}" for text in doc_texts]) if doc_texts else "No valid document content found."

    def process_query(self) -> Message:
        """Process the query and return payment terms as a Message."""
        vector_store = self.build_qdrant()
        docs = vector_store.similarity_search(query=self.query, k=4)
        formatted_docs = self.format_documents([Data(data={"text": doc.page_content, "metadata": doc.metadata}) for doc in docs])

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that reviews contracts and extracts the Payment Terms. "
                "The Payment Terms should be returned in the exact format provided in the example: 'Net 90 Days', "
                "where 'Net' refers to the payment method, and '90 Days' refers to the payment period. "
                "Ensure that the Payment Terms are clearly identified and returned as a string containing only the "
                "payment terms, with no additional text or explanation."
            ),
            ("human", "Query: {query}\nRetrieved Documents:\n{documents}")
        ])
        prompt = prompt_template.format_messages(query=self.query, documents=formatted_docs)

        response = self.llm.invoke(prompt)
        result = Message(text=response.content)
        self.status = result
        return result

    def build(self, query: str) -> str:
        """Method to make the component callable as a tool."""
        self.query = query
        return self.process_query().text