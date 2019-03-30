# Lecture 1 & 2: Word Vectors

### 1. Administration:

**This course hopes to teach:**

- the understanding of the effective modern methods for deep learning used in NLP
- a big picture of understanding human languages and the difficulties in building them
- an understanding of and ability to build NLP systems (in PyTorch), such as explaining word meaning, dependency parsing, machine translation, question answering

**Final project:**

- default: SQuAD

### 2. Human language and human evolutions

*The professor talks about this topic.*

Human languages are a most important invention.Knowledge can be represented in languages and this makes human beings powerful.

**A huge challenge is to represent the meaning of a word:**

1. problem with resources like WordNet: missing nuance, new meanings; subjective, hard to create and adopt; not able to compute easily.
2. traditional NLP (~2012): regards words as discrete symbols like one-hot vectors. **But English words can be countless. More importantly, discrete words share no relationship. (Orthogonal in math terms.)**

These problems lead to distributed semantics.

> You shall know a word by the company it keeps. 

Contexts Matters.

And compared to one-hot vectors, more dense vectors have practical benefits.

### 3. Word2Vec: Overview

Example:



