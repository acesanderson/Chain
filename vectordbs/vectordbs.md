### Vector DBs
- Library_RAG_Chain: 8,000 course titles + descriptions (you can import this and its query_db function as import Course_Descriptions)
- Library_Transcript_Chain: 8,000 course transcripts ('================\n '+ {title: } + '\n================\n' + {transcript})
- Library_TOC_Chain: 8,000 course TOCs (with video descriptions)
- Library_TOC_Chain_Distilled: 8,000 course TOCs (just the chapter + video titles)
- Library_Raptor: Implement hierarchy using clustering and summarization [link](https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb)