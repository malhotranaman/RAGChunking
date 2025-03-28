## RAGChunking
This project implements a retrieval pipeline using an open-source embedding model and evaluates its retrieval quality.

### There are four main elements:
1. Code: Implementations for the chunker and evaluation of the chunking performance
   1. In particular, the class fixed_token_chunker.py and single_corpus_evaluation.py contain these respectively.
2. Experimentation: control.py contains a retrieval evaluation pipeline. 
   1. Init, there is an evaluate function which takes Chunker, Embedding function and Number of retrieved chunks as parameters.
   2. Additionally, experimentation and single-use scripts are provided also. 
3. Dataset: A golden excerpt for a "state of union" transcript and relating questions csv. 
4. Analysis
   1. Analysis of FixedTokenChunker.py walks through all experimentation done in a research format and describes how to best optimize the current chunking strategy.
   2. Experimentation data graphs show visualizations from the results generated. 
   3. Experimentation data is contained in files which contain "experiment" in their title. These data are generated from experimentation scripts in control.

#### Notes
- Citations used in comments are found in the analysis file. 
- questions_df.csv is shortened to only include questions relevant for "state of union". 