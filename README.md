## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Researchers must synthesize findings across many papers quickly. Manually scanning PDFs is slow and error-prone. Build an agentic retrieval system that (1) indexes multiple PDF papers as tool objects, (2) uses a retriever to select the most relevant tools for a query, (3) invokes per-document tools (vector retrieval and summaries) and synthesizes concise, accurate answers, and (4) provides measurable evaluation signals for retrieval and synthesis quality.

### DESIGN STEPS:


### STEP 1: Ingest papers and create per-document tools

Convert each PDF into chunks and embeddings.

Build two tools per document:

vector_tool: performs similarity retrieval over chunks.

summary_tool: returns a compact summary or metadata of the paper.

Ensure tools expose metadata (title, authors, paper id, tool_type) to support retrieval and selection.


### STEP 2: Build a tool-object index and retriever

Collect all tool objects into a single list.

Create an ObjectIndex (backed by a vector index) over the tool objects.

Expose the index as a retriever with similarity_top_k to return the top-k tool objects for a query.


### STEP 3: Instantiate and run the function-calling agent

Create a function-calling agent worker using FunctionCallingAgentWorker.from_tools, passing the tool retriever and an LLM instance.

Provide a system prompt that instructs the agent to always use tools and not rely on external prior knowledge.

Run queries through AgentRunner(agent_worker) to let the agent retrieve tools, call them, and synthesize answers.

Log tool-selection decisions, function calls, and LLM responses for debugging and evaluation.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from utils import get_doc_tools

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]

paper_to_tools_dict = {}
for paper in papers:
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
agent_worker = FunctionCallingAgentWorker.from_tools(initial_tools, llm=llm, verbose=True)
agent = AgentRunner(agent_worker)

resp1 = agent.query("Tell me about the evaluation dataset used in LongLoRA, and then tell me about the evaluation results")
print(str(resp1))

resp2 = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(resp2))
```
### OUTPUT:

<img width="1219" height="710" alt="image" src="https://github.com/user-attachments/assets/2bed82fe-b833-4145-813a-62bc5d22431c" />

<img width="1014" height="418" alt="image" src="https://github.com/user-attachments/assets/b4bf3523-7d42-4cfa-8245-9531e194682a" />

### RESULT:

The multidocument retrieval agent was successfully developed using LlamaIndex, efficiently retrieving, summarizing, and synthesizing information from multiple research papers to produce accurate and concise responses.
