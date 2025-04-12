from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema import Document
import markdown
from bs4 import BeautifulSoup

class MarkdownLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list:
        """Loads and processes a Markdown file into LangChain documents."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        
        # Convert Markdown to HTML
        html_text = markdown.markdown(md_text)

        # Extract text using BeautifulSoup
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text()

        return [Document(page_content=text, metadata={"source": self.file_path})]

def load_markdowns(file_paths: list) -> list:
    """Loads and processes multiple Markdown files into LangChain documents."""
    documents = []
    for file_path in file_paths:
        loader = MarkdownLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
    return documents

def load_texts(file_paths: list) -> list:
    """Loads and processes multiple text files into LangChain documents."""
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
    return documents

def load_pdfs(file_paths: list) -> list:
    """Loads and processes multiple PDF files into LangChain documents."""
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
    return documents