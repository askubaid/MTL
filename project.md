1. Abstract
This project proposes the development of a Multi-Task Learning (MTL) pipeline to simultaneously address two critical aspects of Natural Language Processing i.e Intent Detection and Emotion Classification. While traditional systems treat these tasks independently, this research explores the use of a shared Transformer-based encoder (e.g., RoBERTa ) to leverage the underlying semantic relationship between a user's emotional state and their specific intent. The goal is to create a model that provides a dual-labeled output to better characterize customer interactions in a single inference pass.

2. Research Objectives
• Architectural Design: Implementation of a dual-head Transformer model where a shared encoder backbone feeds into two independent classification heads.
• Joint Optimization: Evaluation of a weighted loss function strategy to optimize the model for both tasks simultaneously.
• Performance Benchmarking: Comparative analysis of the Multi-Task model against independent single-task baselines to assess gains in computational efficiency and classification accuracy.

3. Data & Methodology
• Dataset: The project will utilize a subset of e-commerce customer feedback data (such as the Amazon Customer Reviews or Consumer Complaint datasets). Metadata such as product ratings and categories will be utilized to derive intent labels, while sentiment/emotion markers will serve as the affective labels.
• Framework: The system will be built using the PyTorch python library.
• Hardware: Model training and evaluation will be conducted on my PC RTX 2080 (8GB VRAM) using mixed-precision (FP16) training to ensure local feasibility.

4. Deliverables
• Multi-Task Model: A trained Transformer-based encoder capable of classifying both Intent and Emotion from a single text input.
• Technical Evaluation: A comprehensive report featuring performance metrics (???) and a detailed analysis of the model’s learning curves.
• Inference Script: A functional Python script (CLI-based) to demonstrate the model’s performance on real-time sample inputs for testing.
• A web-based ui for user where user can input review for detecting intent and emotion.
