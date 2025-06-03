# SlicerGPT: Chatbot Extension for 3D Slicer

## Introduction

**SlicerGPT** is an extension for [3D Slicer](https://www.slicer.org/) that integrates a large language model (LLM) directly into the Slicer environment to assist users with 3D Slicer usage.

This extension leverages a **Retrieval-Augmented Generation (RAG)** architecture that combines:

- The active **MRML Scene** (Medical Reality Markup Language), providing the LLM with contextual information about loaded nodes, volumes, and modules in the current session.
- A **FAISS vector store**, which contains:
  - Official Slicer documentation.
  - Python and Markdown files from tutorials, examples, and community extensions.
  - Forum questions and answers, especially from [Slicer Discourse](https://discourse.slicer.org/).

By combining the live scene context with relevant textual knowledge, **SlicerGPT** can generate actionable, context-aware responses to help users navigate and automate 3D Slicer more effectively—even without deep programming experience.

## Requirements

SlicerGPT relies on the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library to run the LLM locally. Therefore, a proper C/C++ compiler is required on your system to build the underlying `llama.cpp` backend.

You can find full installation instructions and platform-specific compiler requirements in the official GitHub repository:  
👉 [https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

Make sure your 3D Slicer's python version is 3.8 or higher.
