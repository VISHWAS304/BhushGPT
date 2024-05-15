# BhushGPT
I built my own large language model, from scratch. Helps us to understand the building , Working and implementation of the Large launguage model.

-----**DAY 01** 5/15/2024-------
1. Created virtual environment named cuda and activated it 
```bash
python -m venv cuda
```
```bash
cuda\Scripts\activate
```
2. Installed certain libraries inside the environment
```bash
pip3 install matplotlib numpy pylzma ipykernel jupyter
```
3. Installed pytorch with cuda
   https://pytorch.org/get-started/locally/  (used this website of pytorch to configure and download according to my system requirements)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
4. Configuring jupyter notebook and its kernel to the cuda kernal created 
```bash
python -m ipykernel install --user --name=cuda --display-name "bhush-gpt" 
```
Launching jupyter notebook
```bash
jupyter notebook
```





























































































# GPT-4 Technical and Mathematical Details (NLP)

| **Aspect**                 | **Details**                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------|
| **Model Type**             | Transformer-based Language Model                                                              |
| **Architecture**           | Decoder-only Transformer                                                                     |
| **Number of Parameters**   | Estimated to be in the hundreds of billions (exact number not publicly disclosed)             |
| **Training Data**          | Diverse text data from the internet, including books, articles, websites, etc.                |
| **Training Objective**     | Minimize cross-entropy loss in predicting the next token in a sequence                        |
| **Tokenization**           | Byte Pair Encoding (BPE)                                                                      |
| **Context Window Size**    | Typically up to 2048 tokens per input (varies based on implementation and use case)           |
| **Activation Function**    | GELU (Gaussian Error Linear Unit)                                                             |
| **Optimization Algorithm** | Adam optimizer                                                                                |
| **Learning Rate Schedule** | Learning rate warm-up followed by linear decay                                                |
| **Batch Size**             | Varies, often in the range of thousands of tokens per batch during training                   |
| **Precision**              | Typically uses mixed precision (FP16 and FP32) during training to balance speed and accuracy  |
| **Fine-tuning Capability** | Can be fine-tuned on specific datasets for domain-specific applications                       |
| **Inference Time**         | Dependent on hardware; typically in milliseconds to seconds per token generation              |
| **Hardware Requirements**  | Requires significant computational resources (GPUs/TPUs) for training and efficient inference |
| **Parallelism**            | Data parallelism and model parallelism used to distribute training across multiple GPUs/TPUs  |
| **Pre-training Time**      | Several weeks to months on large-scale distributed systems                                    |
| **Evaluation Metrics**     | Perplexity, BLEU score, ROUGE score, accuracy on downstream tasks (e.g., QA, summarization)   |
| **Regularization**         | Techniques such as dropout and weight decay used to prevent overfitting                       |
| **Scalability**            | Highly scalable; performance improves with larger datasets and more parameters                |
| **Language Coverage**      | Multilingual support, but performance varies across languages depending on training data      |
| **Bias and Fairness**      | Efforts to mitigate bias, though challenges remain in ensuring fairness and unbiased outputs   |
