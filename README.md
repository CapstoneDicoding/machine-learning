# Machine Learning Path

## Description

Our recommendation system is designed to identify the best candidates and provide candidate ranking based on their compatibility with job requirements. The process is streamlined through several sophisticated steps, ensuring accurate and efficient matching between candidates and job roles.

Our team aims to enhance the HR recruitment process by considering various factors, including:

- **CV Summarization**: Extracted CV texts are processed and summarized using Groq LLM. The summaries are stored in PDF format for a concise overview, aiding in quick decision-making.

- **Candidate Ranking**: Using cosine similarity scores base their embedding (using Sentence Transformer BERT), candidates are ranked. This ranking helps in identifying the top candidates who best meet the job requirements.

- **Recommendation System**: Our system leverages the similarity scores and ranking data to recommend the best candidates for each job role. This ensures that employers receive a curated list of candidates with the highest compatibility, streamlining the hiring process.

By integrating advanced text extraction, processing, and machine learning models, our recommendation system offers a robust solution for finding the best candidates and ranking them effectively. This approach not only enhances the accuracy of candidate-job matching but also significantly improves the efficiency of the recruitment process.

## Workflow Overview

### Dataset Collection
- CVs and Job Requirements: Our process begins by collecting a comprehensive dataset consisting of candidate CVs and detailed job requirements. This data serves as the foundation for our recommendation system. On top of provided CVs and job requirements datasets from Dicoding, we also scraped some job requirements datasets from online website.

### Text Extraction and Processing
- OCR Technology: Using Optical Character Recognition (OCR), we extract the textual content from the CVs. This step is crucial for converting CVs into a machine-readable format.

- Text Processing: The extracted text is then processed and refined. Groq, our base Large Language Model (LLM), is employed to retrieve and structure information from the CV text. The information is stored in JSON format for internal analysis and a summarized version is saved in PDF format for easier review.

### Similarity Scoring
- SentenceTransformer BERT and Cosine Similarity: Using the preprocessed data, the Sentence Tranformer BERT get embeddings of both job requirements and candidate CVs. The cosine similarity metric is then applied to compute a similarity score, indicating how well each candidate's CV matches the job requirements.
