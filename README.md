CASCaRA GraphRAG v0.3

Simple setup and run instructions.

Dependencies

Install:

Python 3.10+

Neo4j Desktop

APOC plugin

Graph Data Science (GDS) plugin

Ollama

Install Python packages

Run:

pip install neo4j 

pip install ollama

pip install python-dotenv

Neo4j plugins

In Neo4j Desktop, open your database go to plugins and install:

APOC

Graph Data Science

you will need to restart the DB after installing these

Verify APOC:

RETURN apoc.version();

Verify GDS:

RETURN gds.version();

Ollama models

Install the embedding model:

ollama pull qwen3-embedding:0.6b

Install the chat model:

ollama pull qwen3:8b

Check installed models:

ollama list

Fill out the env witht he neo4j input and parameters.

run embedding.py once

after its done embedding run GraphRag.py than ask it questions about the Graph data.