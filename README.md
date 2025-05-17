# Joint Mid-Fusion Attention Mechanism for Multimodal Argumentative Fallacy Detection with Reduced Biases

MMUSED Fallacy 2024 Data Splits from Link: [Download](https://drive.google.com/drive/folders/1uY35jfiKZnsCAvJppCe7NMJnIssjWG2k?usp=sharing)


Tasks: 
- Argumentative Fallacy Detection (AFD)
- Argumentative Fallacy Classification (AFC)

#### Joint Mid-Fusion Attention

![image](https://github.com/user-attachments/assets/6e115ee7-218b-4d46-b1f5-993d7e84377b)


#### Example Data Bias Graphs:
![AFD_FREQUENT](https://github.com/user-attachments/assets/7c1016cd-0690-4784-8bea-063ae7d6fbf2)
![AFD_LABELS](https://github.com/user-attachments/assets/4bab1fcb-e5bd-41bf-adda-4d4b87b4bb20)
![AFC_TFIDF_AVG](https://github.com/user-attachments/assets/ff2d8596-c7a5-4bdb-8d65-7c91d2589ae0)
![AFC_LABELS](https://github.com/user-attachments/assets/53ea292b-e8ce-483c-92f4-7dafb4ebc9e5)


**Note**: Significant breaking modifications have been made to the original MAMKit source code, specifically in, dataset processors, dataset zenodo links and data collators in response to the persistent update issues encountered within the framework during March and April 2025. To ensure reproducibility and stability, users are advised to avoid relying on the latest upstream updates and instead utilize the locally adapted source files and datasets provided here. These files not only address compatibility concerns but also include additional implementations for the proposed joint attention architecture, cross-modal attention visualization, and dataset bias analysis modules. The original baseline benchmark files are also included for reference and comparison.

**Acknowledgement**

We thank the [MMUsed Fallacy Task](https://nlp-unibo.github.io/mm-argfallacy/2025/) Organizers for developing [MAMKit](https://nlp-unibo.github.io/mamkit/): Multimodal Argument Mining Toolkit and providing an easy to extend interface with PyTorch for Fallacy Dataset Collators and Baseline Benchmarks.

Eleonora Mancini, Federico Ruggeri, Stefano Colamonaco, Andrea Zecca, Samuele Marro, and Paolo Torroni. 2024. MAMKit: A Comprehensive Multimodal Argument Mining Toolkit. In Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024), pages 69–82, Bangkok, Thailand. Association for Computational Linguistics.
