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
    "13578_Diffusion_Based_Planning.pdf",
    "13985_RM_Bench_Benchmarking_Re.pdf",
    "14257_DarkBench_Benchmarking_D.pdf",
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

resp1 = agent.query("Summarize the key contributions of Diffusion_Based_Planning.")
print(str(resp1))

resp2 = agent.query("Explain the methodology and architecture used in RM_Bench.")

print(str(resp2))

resp3= agent.query("What evaluation metrics were used in DarkBench, and what were the main findings?")
print(str(resp3))

```
### OUTPUT:

```

Added user message to memory: Summarize the key contributions of Diffusion_Based_Planning.
=== Calling Function ===
Calling function: summary_tool_13578_Diffusion_Based_Planning with args: {"input": "key contributions"}
=== Function Output ===
The key contributions of the research discussed include harnessing diffusion models with a specifically designed architecture for high-performance motion planning, achieving state-of-the-art performance on real-world datasets, demonstrating personalized driving behavior at runtime, collecting and evaluating a new delivery-vehicle dataset, redefining the planning task as a future trajectory generation task, introducing the Diffusion Planner for enhanced autonomous planning, showcasing practical implementations for closed-loop planning, utilizing classifier guidance for driving behavior alignment, and providing a training-free approach for trajectory customization.
=== LLM Response ===
The key contributions of Diffusion_Based_Planning include harnessing diffusion models with a specially designed architecture for high-performance motion planning, achieving top performance on real-world datasets, demonstrating personalized driving behavior at runtime, collecting and evaluating a new delivery-vehicle dataset, redefining the planning task as a future trajectory generation task, introducing the Diffusion Planner for improved autonomous planning, showcasing practical implementations for closed-loop planning, using classifier guidance for driving behavior alignment, and offering a training-free approach for trajectory customization.
assistant: The key contributions of Diffusion_Based_Planning include harnessing diffusion models with a specially designed architecture for high-performance motion planning, achieving top performance on real-world datasets, demonstrating personalized driving behavior at runtime, collecting and evaluating a new delivery-vehicle dataset, redefining the planning task as a future trajectory generation task, introducing the Diffusion Planner for improved autonomous planning, showcasing practical implementations for closed-loop planning, using classifier guidance for driving behavior alignment, and offering a training-free approach for trajectory customization.
Added user message to memory: Explain the methodology and architecture used in RM_Bench.
=== Calling Function ===
Calling function: vector_tool_13985_RM_Bench_Benchmarking_Re with args: {"query": "methodology and architecture in RM_Bench"}
=== Function Output ===
The methodology in RM-Bench involves constructing a benchmark for evaluating reward models that focuses on subtlety and style. It includes experiments to demonstrate a strong correlation with policy model performance. The architecture of RM-Bench consists of using reward models designed to provide reward signals based on specific preferences, typically constructed upon large pre-trained language models by adding a classification head to predict the reward of a response given a prompt. The benchmark aims to authentically reflect the performance of reward models and establish a high correlation with policy model performance, serving as a reliable reference for selecting reward models for language model alignment.
=== LLM Response ===
The methodology in RM-Bench focuses on constructing a benchmark for evaluating reward models with a focus on subtlety and style. It includes experiments to demonstrate a strong correlation with policy model performance. 

The architecture of RM-Bench involves using reward models designed to provide reward signals based on specific preferences. These models are typically constructed upon large pre-trained language models by adding a classification head to predict the reward of a response given a prompt. 

Overall, RM-Bench aims to authentically reflect the performance of reward models and establish a high correlation with policy model performance. It serves as a reliable reference for selecting reward models for language model alignment.
assistant: The methodology in RM-Bench focuses on constructing a benchmark for evaluating reward models with a focus on subtlety and style. It includes experiments to demonstrate a strong correlation with policy model performance. 

The architecture of RM-Bench involves using reward models designed to provide reward signals based on specific preferences. These models are typically constructed upon large pre-trained language models by adding a classification head to predict the reward of a response given a prompt. 

Overall, RM-Bench aims to authentically reflect the performance of reward models and establish a high correlation with policy model performance. It serves as a reliable reference for selecting reward models for language model alignment.
Added user message to memory: What evaluation metrics were used in DarkBench, and what were the main findings?
=== Calling Function ===
Calling function: vector_tool_14257_DarkBench_Benchmarking_D with args: {"query": "evaluation metrics"}
=== Function Output ===
The evaluation metrics used in the study included assessing the occurrence of dark pattern instances across all categories, identifying the most commonly occurring dark patterns, and analyzing the variance in rates of different dark patterns among the models tested on the DarkBench benchmark.
=== Calling Function ===
Calling function: summary_tool_14257_DarkBench_Benchmarking_D with args: {"input": "main findings"}
=== Function Output ===
The main findings from the provided context encompass the detection of dark design patterns in interactions with large language models (LLMs) through the introduction of the DarkBench benchmark. The study revealed manipulative behaviors in some LLMs, such as favoring developers' products, untruthful communication, and other deceptive practices. Variations in the occurrence of dark patterns were observed between models, with consistency within models from the same developer. Models from companies emphasizing safety and ethical standards displayed lower rates of dark patterns. Sneaking was identified as the most common dark pattern, while sycophancy was the least common. Discussions also covered benchmarking cyber capabilities, training ethical AI assistants, deceptive patterns in writing assistants, ethics of the attention economy, and the impact of post-training enhancements on AI systems. Additionally, research explored echo chambers created by large language models, the introduction of various language models by OpenAI, training models to follow instructions with human feedback, trade-offs between rewards and ethical behavior, and the impact of language models on diverse information seeking. Studies also delved into the potential deceptive behavior of LLMs, their strategic deception of users, and the prevalence and prevention of their use in crowd work. Furthermore, discussions included dark patterns in chatbot categories, comparisons between annotator model families based on metrics like anthropomorphization, brand bias, harmful content, and generation, as well as evaluations of different language models across metrics such as anthropomorphization, brand bias, harmful generation, sneaking, sycophancy, and user retention. These findings shed light on how LLMs handle various prompts related to social interactions, brand preferences, content manipulation, emotional support, and generation of potentially harmful content, revealing strengths and weaknesses in different scenarios.
=== LLM Response ===
The evaluation metrics used in DarkBench included assessing the occurrence of dark pattern instances across all categories, identifying the most commonly occurring dark patterns, and analyzing the variance in rates of different dark patterns among the models tested on the DarkBench benchmark.

The main findings from DarkBench encompass the detection of dark design patterns in interactions with large language models (LLMs) through the introduction of the DarkBench benchmark. The study revealed manipulative behaviors in some LLMs, such as favoring developers' products, untruthful communication, and other deceptive practices. Variations in the occurrence of dark patterns were observed between models, with consistency within models from the same developer. Models from companies emphasizing safety and ethical standards displayed lower rates of dark patterns. Sneaking was identified as the most common dark pattern, while sycophancy was the least common. 

The research also covered benchmarking cyber capabilities, training ethical AI assistants, deceptive patterns in writing assistants, ethics of the attention economy, and the impact of post-training enhancements on AI systems. Additionally, discussions included echo chambers created by large language models, the introduction of various language models by OpenAI, training models to follow instructions with human feedback, trade-offs between rewards and ethical behavior, and the impact of language models on diverse information seeking. Studies also delved into the potential deceptive behavior of LLMs, their strategic deception of users, and the prevalence and prevention of their use in crowd work. Furthermore, discussions included dark patterns in chatbot categories, comparisons between annotator model families based on metrics like anthropomorphization, brand bias, harmful content, and generation, as well as evaluations of different language models across metrics such as anthropomorphization, brand bias, harmful generation, sneaking, sycophancy, and user retention. These findings shed light on how LLMs handle various prompts related to social interactions, brand preferences, content manipulation, emotional support, and generation of potentially harmful content, revealing strengths and weaknesses in different scenarios.
assistant: The evaluation metrics used in DarkBench included assessing the occurrence of dark pattern instances across all categories, identifying the most commonly occurring dark patterns, and analyzing the variance in rates of different dark patterns among the models tested on the DarkBench benchmark.

The main findings from DarkBench encompass the detection of dark design patterns in interactions with large language models (LLMs) through the introduction of the DarkBench benchmark. The study revealed manipulative behaviors in some LLMs, such as favoring developers' products, untruthful communication, and other deceptive practices. Variations in the occurrence of dark patterns were observed between models, with consistency within models from the same developer. Models from companies emphasizing safety and ethical standards displayed lower rates of dark patterns. Sneaking was identified as the most common dark pattern, while sycophancy was the least common. 

The research also covered benchmarking cyber capabilities, training ethical AI assistants, deceptive patterns in writing assistants, ethics of the attention economy, and the impact of post-training enhancements on AI systems. Additionally, discussions included echo chambers created by large language models, the introduction of various language models by OpenAI, training models to follow instructions with human feedback, trade-offs between rewards and ethical behavior, and the impact of language models on diverse information seeking. Studies also delved into the potential deceptive behavior of LLMs, their strategic deception of users, and the prevalence and prevention of their use in crowd work. Furthermore, discussions included dark patterns in chatbot categories, comparisons between annotator model families based on metrics like anthropomorphization, brand bias, harmful content, and generation, as well as evaluations of different language models across metrics such as anthropomorphization, brand bias, harmful generation, sneaking, sycophancy, and user retention. These findings shed light on how LLMs handle various prompts related to social interactions, brand preferences, content manipulation, emotional support, and generation of potentially harmful content, revealing strengths and weaknesses in different scenarios.

​```
### RESULT:

The multidocument retrieval agent was successfully developed using LlamaIndex, efficiently retrieving, summarizing, and synthesizing information from multiple research papers to produce accurate and concise responses.

