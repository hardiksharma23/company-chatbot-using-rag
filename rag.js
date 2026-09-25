/*
Implementation

Stage-1 : indexting

1. Load the document
2. Chunks the document
3. generate the embeddings  ---> free use using langchain
4. store the vector embeddings --- vector database ---> pinecone free tier

Stage-2 : usingg the chatbot

1. Set up the LLM
2. Retrival Step
3. Pass input + relevant information to LLM


*/

import { indexTheDocument } from "./prepare.js";

const filePath = './cg-internal-docs.pdf'
indexTheDocument(filePath);