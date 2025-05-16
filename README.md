# Joint Mid-Fusion Attention Mechanism for Multimodal Argumentative Fallacy Detection with Reduced Biases

MMUSED Fallacy 2024 Data Splits from Link: [Download](https://drive.google.com/drive/folders/1uY35jfiKZnsCAvJppCe7NMJnIssjWG2k?usp=sharing)


Tasks: 
- Argumentative Fallacy Detection (AFD)
- Argumentative Fallacy Classification (AFC)




Note: Significant breaking modifications have been made to the original MAMKit source code, specifically in, dataset processors, dataset zenodo links and data collators in response to the persistent update issues encountered within the framework during March and April 2025. To ensure reproducibility and stability, users are advised to avoid relying on the latest upstream updates and instead utilize the locally adapted source files and datasets provided here. These files not only address compatibility concerns but also include additional implementations for the proposed joint attention architecture, cross-modal attention visualization, and dataset bias analysis modules. The original baseline benchmark files are also included for reference and comparison.

Acknowledgement

We thank the [MMUsed Fallacy Task](https://nlp-unibo.github.io/mm-argfallacy/2025/) Organizers for developing [MAMKit](https://nlp-unibo.github.io/mamkit/): Multimodal Argument Mining Toolkit and providing an easy to extend interface with PyTorch for Fallacy Dataset Collators and Baseline Benchmarks.
