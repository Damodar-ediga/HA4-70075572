# NLP and AI Assignment - Damodar Goud Ediga

## Introduction

This assignment demonstrates the implementation of various Natural Language Processing (NLP) techniques such as text preprocessing, named entity recognition, scaled dot-product attention, and sentiment analysis. The solutions utilize libraries like SpaCy and HuggingFace Transformers to solve real-world problems in NLP. 

### Student Information
- **Name**: Damodar Goud Ediga
- **ID**: 700755572

---

## Task 1: NLP Preprocessing Pipeline

### Explanation:
In this task, we create a function to preprocess a sentence using several steps common in NLP pipelines. These steps are:

1. **Tokenization**: Splitting the input sentence into individual tokens (words or punctuation marks).
2. **Stopword Removal**: Removing common words like "the", "and", "is", etc., which usually don't carry important meaning in the context of NLP tasks.
3. **Stemming**: Reducing each word to its root form (e.g., "running" becomes "run") using a stemming algorithm.

The function is applied to the sentence: *"NLP techniques are used in virtual assistants like Alexa and Siri."* It prints the tokens before and after removing stopwords and after stemming, showcasing how these preprocessing steps transform the data.

### Key Concepts:
- **Tokenization**: Breaking the text into smaller units like words.
- **Stopword Removal**: Eliminating common words that do not add significant meaning.
- **Stemming**: Reducing words to their root form to standardize different word variations.

---

## Task 2: Named Entity Recognition (NER) with SpaCy

### Explanation:
In this task, we use the SpaCy library to perform Named Entity Recognition (NER) on a sentence. The goal of NER is to identify and classify entities such as names of people, organizations, dates, and other proper nouns.

The function processes the sentence: *"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."* It extracts named entities (e.g., "Barack Obama", "United States", "2009") and prints:
- The entity text (e.g., "Barack Obama")
- The entity label (e.g., "PERSON")
- The start and end character positions in the original text.

### Key Concepts:
- **Named Entity Recognition (NER)**: Identifying proper nouns and categorizing them (e.g., people, locations, dates).
- **Entity Labels**: Tags such as PERSON, GPE (Geopolitical Entity), DATE, etc., indicating the type of the recognized entity.

---

## Task 3: Scaled Dot-Product Attention

### Explanation:
This task focuses on implementing the **scaled dot-product attention** mechanism, which is a fundamental component of attention models like those used in Transformers. The function computes attention scores using the Query (Q), Key (K), and Value (V) matrices. It follows these steps:

1. Compute the dot product between the Query (Q) and Key (Kᵀ) matrices.
2. Scale the result by dividing by the square root of the dimensionality of the Key matrix (√d).
3. Apply the **softmax function** to convert the scaled scores into attention weights.
4. Multiply the attention weights by the Value (V) matrix to produce the final output.

### Key Concepts:
- **Dot-Product**: Measures the similarity between the Query and Key matrices.
- **Scaling by √d**: Helps to prevent extremely large values in the attention scores as the dimensionality of Q and K increases.
- **Softmax**: Converts raw scores into a probability distribution (attention weights).
- **Attention Weights**: Determine the importance of each word or token in the context of the others.

---

## Task 4: Sentiment Analysis using HuggingFace Transformers

### Explanation:
For this task, we leverage the HuggingFace **Transformers** library to perform sentiment analysis on a given text. The function uses a pre-trained sentiment analysis model and takes the input sentence: *"Despite the high price, the performance of the new MacBook is outstanding."* The model outputs:
- **Sentiment Label** (e.g., "POSITIVE" or "NEGATIVE")
- **Confidence Score** (e.g., a value between 0 and 1 indicating the model's confidence in the prediction)

By using a pre-trained model, we avoid the need to train the model from scratch, benefiting from the knowledge it has gained during training on vast datasets.

### Key Concepts:
- **Sentiment Analysis**: Identifying the sentiment (positive, negative, neutral) of a piece of text.
- **Pre-trained Models**: Models that have been trained on large datasets and can be fine-tuned for specific tasks without requiring complete retraining.

---

## Conclusion

This assignment demonstrates several core NLP techniques that are essential for real-world applications. From basic preprocessing to advanced tasks like named entity recognition and sentiment analysis, the solutions provided showcase the power of modern NLP libraries. These techniques form the foundation for building more complex models in various NLP-based tasks.

---

All  questions are implemented in a single Google Colab notebook.

Google Colab Notebook File Name : HA4.ipynb
-----------------------------------------
You can either run it locally or open it in Colab: https://colab.research.google.com/drive/1xiDzzcJF2-OB3zxLNERcsWTMqOFdgQoq?usp=sharing
-----------------------------------------
VIDEO LINK: https://drive.google.com/file/d/1PteOFn_y3MdZdYjJMIcFWw5O1SqX16EK/view?usp=share_link

-----------------------------------------


