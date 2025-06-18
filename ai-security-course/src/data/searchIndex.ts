// This file is auto-generated. Do not edit manually.
// Generated on: 2025-06-18T14:29:46.418Z

export interface SearchIndexEntry {
  title: string;
  content: string;
  sectionTitle: string;
  sectionId: string;
}

export type SearchIndex = Record<string, SearchIndexEntry>;

export const searchIndex: SearchIndex = {
  "/": {
    "title": "Installation",
    "content": "1.1. Installation Setting up a secure AI development environment is the first step in building secure AI systems. This guide will walk you through the installation process for the UChicago XLab AI Security toolkit. The xlab-security Python package is the best way to get started with AI security development. This package provides essential tools and helper functions for building secure AI systems. Prerequisites Make sure you have Python 3.7 or higher installed on your system. You can check your Python version by running: Installation from PyPI Once the package is published to PyPI, you can install it directly: Installation from TestPyPI (for testing) For testing the latest development version, you can install from TestPyPI: Verify Installation After installation, you can verify that the package is working correctly: This should output something like: Development Installation If you're contributing to the package or want to install from source: 1.1.1. Update existing AI security environment You can check your current package version: To update to the latest version:",
    "sectionTitle": "Getting Started",
    "sectionId": "1"
  },
  "/model-inference-attacks/stealing-model-weights": {
    "title": "Model Extraction Attacks",
    "content": "Introduction to Model Stealing Techniques This section introduces practical techniques for model extraction attacks - a significant concern in AI security. When deploying AI models, particularly large language models (LLMs), organizations must be aware that even black-box access to models can leak information about their architecture and parameters. Learning Objectives By the end of this section, you will: - Understand how to run a GPT-2 model locally for experimentation - Learn how to extract a model's hidden dimension size from its outputs - Understand the mathematical principles behind model extraction attacks - Recognize the security implications of these vulnerabilities Mathematical Intuition The core insight behind model extraction attacks comes from understanding the architecture of transformer-based language models. In these models: - The final layer projects from a hidden dimension h to vocabulary size l - This creates a mathematical bottleneck where output logits can only span a subspace of dimension h - By collecting many output vectors and analyzing their singular values, we can determine this hidden dimension Mathematically, when a language model processes text: $$f\\theta(p) = \\text{softmax}(\\mathbf{W} \\cdot g\\theta(p))$$ Where: - $\\mathbf{W}$ is an $l \\times h$ matrix (vocabulary size × hidden dimension) - $g\\theta(p)$ outputs an $h$-dimensional hidden state vector This means that no matter how many different inputs we try, the rank of the output logit matrix cannot exceed h_. This property allows us to extract proprietary information about model architecture through careful analysis. Hands-on Exercise This concept is demonstrated in the accompanying Jupyter notebook, which shows: 1. How to run GPT-2 locally and generate text 2. How temperature affects text generation (preventing repetition) 3. How to implement the model extraction attack described in \"Stealing Part of a Production Language Model\" (Carlini et al., 2024) In the practical exercise, you'll: - Generate random prefixes to query the model - Collect logit vectors from model outputs - Apply Singular Value Decomposition (SVD) to determine the hidden dimension - Visualize the \"cliff edge\" in singular values that reveals the model's dimension Security Implications This type of attack demonstrates that: 1. Even black-box access to models can leak architectural details 2. Proprietary information about model design can be extracted through API calls 3. Knowledge of model dimensions enables more sophisticated attacks 4. Traditional API security measures may not protect against these mathematical vulnerabilities Defensive Considerations To protect against model extraction attacks, consider: - Limiting the precision of model outputs - Adding controlled noise to model responses - Implementing rate limiting and monitoring for suspicious query patterns - Using watermarking techniques to detect model stealing attempts Notebook Access The complete code for this exercise is available in the Model Extraction Notebook where you can run the code yourself and experiment with different parameters. Further Reading - Stealing Part of a Production Language Model by Carlini et al. (2024) - Extracting Training Data from Large Language Models by Carlini et al. (2021) - Membership Inference Attacks on Machine Learning Models by Shokri et al. (2017)",
    "sectionTitle": "Model Extraction",
    "sectionId": "3"
  }
};

export default searchIndex;
