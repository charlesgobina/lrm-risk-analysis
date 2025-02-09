import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from dataclasses import dataclass, asdict
from anthropic import Anthropic
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_cpp import Llama


# Load environment variables
load_dotenv()

class Lrm:
    def __init__(self, model_path: str, n_ctx: int, chat_format: str, main_gpu: int):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.chat_format = chat_format
        self.main_gpu = main_gpu
    
    def __repr__(self):
        return f"Llama(model_path={self.model_path}, n_ctx={self.n_ctx}, chat_format={self.chat_format}, main_gpu={self.main_gpu})"
    

@dataclass
class ExcelChunk:
    """Represents a logical chunk of Excel data with metadata"""
    content: str
    sheet_name: str
    row_start: int
    row_end: int
    columns: List[str]
    metadata: Dict[str, Any]

class ExcelProcessor:
    """Handles Excel file processing and chunking"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.xlsx = pd.ExcelFile(file_path)
    
    def _clean_text(self, text: Any) -> str:
        """Clean and standardize text values"""
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    def _process_merged_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle merged cells while preserving relationships"""
        for col in df.columns:
            df[col] = df[col].fillna(method='ffill')
        return df
    
    def _create_chunk_content(self, chunk_df: pd.DataFrame) -> str:
        """Create a structured text representation of the chunk"""
        content_parts = []
        
        # Add header information
        headers = {col: chunk_df[col].iloc[0] for col in chunk_df.columns 
                  if chunk_df[col].nunique() == 1}
        if headers:
            content_parts.append("Context:")
            for col, val in headers.items():
                content_parts.append(f"{col}: {self._clean_text(val)}")
        
        # Add data rows
        content_parts.append("\nData:")
        for _, row in chunk_df.iterrows():
            row_parts = []
            for col in chunk_df.columns:
                value = self._clean_text(row[col])
                if value and value not in str(headers.get(col, '')):
                    row_parts.append(f"{col}: {value}")
            if row_parts:
                content_parts.append(" | ".join(row_parts))
        
        return "\n".join(content_parts)
    
    def process_sheet(self, sheet_name: str, chunk_size: int = 10) -> List[ExcelChunk]:
        """Process a single sheet into chunks"""
        df = pd.read_excel(self.xlsx, sheet_name=sheet_name)
        df = self._process_merged_cells(df)
        
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            content = self._create_chunk_content(chunk_df)
            
            chunk = ExcelChunk(
                content=content,
                sheet_name=sheet_name,
                row_start=i,
                row_end=min(i + chunk_size, len(df)),
                columns=list(df.columns),
                metadata={
                    "file_name": Path(self.file_path).name,
                    "total_rows": len(df)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_all_sheets(self, chunk_size: int = 10) -> List[ExcelChunk]:
        """Process all sheets in the Excel file"""
        all_chunks = []
        for sheet_name in self.xlsx.sheet_names:
            sheet_chunks = self.process_sheet(sheet_name, chunk_size)
            all_chunks.extend(sheet_chunks)
        return all_chunks

class VectorStore:
    """Manages the vector database operations"""
    
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="excel_documents",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _generate_chunk_id(self, chunk: ExcelChunk) -> str:
        """Generate a unique ID for a chunk"""
        unique_string = f"{chunk.sheet_name}_{chunk.row_start}_{chunk.row_end}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def add_chunks(self, chunks: List[ExcelChunk], batch_size: int = 100):
        """Add chunks to the vector store"""
        current_batch = {"ids": [], "documents": [], "metadatas": []}
        
        for chunk in chunks:
            chunk_id = self._generate_chunk_id(chunk)
            document = f"Sheet: {chunk.sheet_name}\n{chunk.content}"
            metadata = {
                "sheet_name": chunk.sheet_name,
                "row_start": chunk.row_start,
                "row_end": chunk.row_end,
                "columns": ",".join(chunk.columns),
                **chunk.metadata
            }
            
            current_batch["ids"].append(chunk_id)
            current_batch["documents"].append(document)
            current_batch["metadatas"].append(metadata)
            
            if len(current_batch["ids"]) >= batch_size:
                self.collection.add(**current_batch)
                current_batch = {"ids": [], "documents": [], "metadatas": []}
        
        if current_batch["ids"]:
            self.collection.add(**current_batch)
    
    def search(self, query: str, n_results: int = 5, 
               sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        where = {"sheet_name": sheet_name} if sheet_name else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        formatted_results = []
        for idx in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][idx],
                'metadata': results['metadatas'][0][idx],
                'distance': results['distances'][0][idx]
            })
        
        return formatted_results

class RAGSystem:
    """Main RAG system combining all components"""
    
    def __init__(self, 
                 persist_dir: str = "./rag_db",
                 anthropic_api_key: Optional[str] = None):
        self.vector_store = VectorStore(persist_dir)
        self.lrm = Lrm(
            model_path="models/O1-OPEN/OpenO1-LLama-8B-v0.1",
            n_ctx=4096,
            chat_format="qwen",
            main_gpu=0
        )
        self.anthropic_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
    
    def index_excel(self, file_path: str, chunk_size: int = 10):
        """Process and index an Excel file"""
        processor = ExcelProcessor(file_path)
        chunks = processor.process_all_sheets(chunk_size)
        self.vector_store.add_chunks(chunks)
        
        print(f"Indexed {len(chunks)} chunks from {file_path}")
    
    def generate_prompt(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate a prompt for the LLM"""
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        return f"""Based on the following Excel data:

          {context}

          Question: {query}

          You are an expert in threat and vulnerability analysis. Given a scenario in the question above containing details about a threat and its associated vulnerabilities, analyze the scenario thoroughly and provide the following information:

1. **Predicted Remediation Strategy(ies):** Suggest the most effective countermeasure(s) to address the identified threat and vulnerability.  
2. **Reasoning:** Provide a concise explanation for why this remediation strategy is appropriate based on the context.  
3. **Classification Description:** Describe how you classified this threat-vulnerability pair (e.g., based on impact, likelihood, or other criteria).  
4. **Threat ID:** Identify the corresponding Threat ID from the provided data.  
5. **Vulnerability ID:** Identify the corresponding Vulnerability ID from the provided data.  
6. **Remediation ID (Countermeasure ID):** Provide the relevant countermeasure ID(s) that align with the suggested remediation strategy.

### **Context Format:**  
- **Threat:** [Threat Name]  
- **Vulnerability:** [Vulnerability Description]  
- **Additional Details:** [Impact level, relationships, etc.]  
- **Available Countermeasures:** [List of possible countermeasures from the dataset]  

### **Example Input:**  
- **Threat:** Flooding  
- **Vulnerability:** Inadequate Flood Protection  
- **Impact Level:** 3  
- **Countermeasures:**  
  - f8.4: Flood Control with Sensors  
  - f12.1: Automatic Water System Shutdown  
  - f12.4: Anti-Flooding Pumps  

### **Expected Output Format:**  
- **Predicted Remediation Strategy:** Anti-Flooding Pumps (f12.4), Automatic Water System Shutdown (f12.1)  
- **Reasoning:** Anti-flooding pumps actively mitigate flooding risks, while automatic shutdown reduces damage from water exposure.  
- **Classification Description:** Environmental threat with infrastructure vulnerability, classified based on high impact potential.  
- **Threat ID:** V27  
- **Vulnerability ID:** V27  
- **Remediation ID:** f12.1, f12.4  

Ensure the output is structured and concise, making it easy to parse for downstream processing. Base all responses strictly on the provided data and avoid speculative answers."""
              
    def query(self, 
              query: str, 
              n_results: int = 5,
              sheet_name: Optional[str] = None,
              return_sources: bool = False) -> Dict[str, Any]:
        """Query the system and generate a response"""
        # Get relevant chunks
        relevant_chunks = self.vector_store.search(
            query=query,
            n_results=n_results,
            sheet_name=sheet_name
        )
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the Excel data.",
                "sources": [] if return_sources else None
            }

        print(relevant_chunks)


        
        # Generate answer using LRM
        prompt = self.generate_prompt(query, relevant_chunks)
        qwens = Llama(
            model_path="models/llama-o1-supervised-1129-q4_k_m.gguf",
            n_ctx=4096,
            chat_format="qwen",
            main_gpu=0
        )
        response = qwens.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an AI assistant helping with risk analysis of threats and vulnerabilities in a mission-critical system. Provide responses based on the given context without fabricating information. Your response should be well formatted, including : threat ID, threat name, description, vulnerability ID, name, description, countermeasure ID, and your reasoning."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in response:
            print(chunk['choices'][0]['delta'].get('content', ''), end='', flush=True)

        





        # if self.anthropic_client:
        #     prompt = self.generate_prompt(query, relevant_chunks)
        #     response = self.anthropic_client.messages.create(
        #         model="claude-3-sonnet-20240229",
        #         max_tokens=1000,
        #         messages=[{
        #             "role": "user",
        #             "content": prompt
        #         }]
        #     )
        #     answer = response.content[0].text
        # else:
        #     # If no API key provided, return relevant chunks as response
        #     answer = "API key not provided. Here are the relevant excerpts:\n\n" + \
        #             "\n\n".join([f"From {chunk['metadata']['sheet_name']}:\n{chunk['content']}" 
        #                         for chunk in relevant_chunks])
        
        return {
            # "answer": answer,
            "sources": relevant_chunks if return_sources else None
        }


# Example usage
def main():
    # Initialize the RAG system
    rag = RAGSystem(
        persist_dir="./excel_rag_db",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Index an Excel file
    rag.index_excel("test_data/ra.xlsx", chunk_size=10)
    
    # Example queries
    queries = [
        "what threat is associated with Network Worm"
    ]
    
    for query in queries:
        print(f"\nQuestion: {query}")
        result = rag.query(query, n_results=3, return_sources=True)
        
        print("\nAnswer:")
        # print(result["answer"])
        
        if result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. From sheet '{source['metadata']['sheet_name']}'")
                print(f"Rows {source['metadata']['row_start']}-{source['metadata']['row_end']}")
                print(f"Relevance score: {1 - source['distance']:.4f}")

if __name__ == "__main__":
    main()