Multi-Task Learning Pipeline (Intent & Emotion) - Implementation Plan
This document outlines the proposed plan for developing a Multi-Task Learning (MTL) system using a Transformer-based encoder to simultaneously classify a user's intent and emotion from textual feedback. The plan covers custom dataset generation, model training, evaluation, and a full-stack web-based user interface.

Proposed Changes & Milestones
1) Dataset creation and validation pipeline
Dataset Sourcing: Identify two separate datasets from Hugging Face (e.g., one for customer intents, and one for emotions like go_emotions).
Logical Combination Pipeline: Develop a Python script (create_dataset.py) to map, merge, or logically combine these two datasets so that each text entry is assigned both a valid Intent label and an Emotion label.
Data Validation Script: Create a script (validate_dataset.py) to check the health of the generated dataset (e.g., checking class distributions, identifying missing labels, and validating the overall structure).

2) Selecting & training the model on new dataset
Model Architecture: Implement a MultiTaskRoBERTa PyTorch module using a shared roberta-base encoder that branches into two separate linear classification heads (Intent and Emotion).
Dataloaders: Load the newly generated and validated custom dataset, utilizing a PyTorch Dataset class and DataLoader.
Joint Optimization: Implement a training loop with a joint loss function (e.g., a weighted sum of Cross-Entropy Loss for both tasks). We will use PyTorch's Automatic Mixed Precision (AMP/FP16) to ensure the model trains efficiently on an 8GB RTX 2080.
Checkpoints: Track validation loss and save the best-performing model weights.
3) Inference loop & finding what benchmarking metrics we should use
Benchmarking Metrics: We will evaluate the multi-task model using independent metrics for both heads: Accuracy and Macro F1-score. We'll also compute a combined metric (e.g., average Macro F1) to compare against single-task baselines if needed.
Learning Curves: Generate and save plots for training and validation loss/accuracy across epochs to analyze convergence.
Inference CLI: Develop a script (inference.py) that loads the model weights, accepts a raw string review from the terminal, and outputs the predicted Intent and Emotion.

Verification Plan
Automated Tests
The dataset validation script (validate_dataset.py) will automatically flag anomalies in the generated dataset prior to training.
The training loop will validate the model against a holdout test set to ensure the dual-head architecture achieves competitive accuracy.
Manual Verification
We will test the CLI inference script manually with out-of-distribution e-commerce reviews to qualitatively assess the model's predictions.
We will run the Web UI locally, testing the client-server interaction to ensure accurate predictions are displayed, and visually verify the design aesthetics and responsiveness.

4) Web UI for end user
Folder Structure: Strictly separate the application into two distinct directories: backend/ and frontend/.
Backend API (backend/): Develop a Python FastAPI application to load the trained PyTorch model and serve predictions via a REST endpoint (/predict).
Frontend App (frontend/): Build a modern, highly aesthetic UI using Vite (React). This will provide a sleek, premium experience where users can input reviews and view dynamic animations of the predicted intent and emotion.


5) Milestone 5 will be to push all the code to gihub repository, https://github.com/askubaid/MTL.git, and then we have to deploy our backend local machine and frontend part on Github Pages. because the model will run on local machine so we have look for ngrok to expose our local backend to the internet. So the frontend will call the backend using ngrok url to get the predictions.

    Milestone 5 will have 2 parts.

    Part A:  Upload the repository to github and deploy front end to github pages. 

    Part B: Deploy backend on local machine and run it on ngrok to expose it to the internet. So the frontend will call the backend using ngrok url to get the predictions. 

We will do the A part first and then do the Part B


