# Evaluating Deep Learning Challenges in Lightweight AES Cryptanalysis

## Overview
This repository contains the implementation and dataset used for the research paper **"Evaluating Deep Learning Challenges in Lightweight AES Cryptanalysis"**, supported by Nile University. The study explores the application of deep learning models in cryptanalysis, specifically focusing on the Simplified Advanced Encryption Standard (S-AES) in Counter (CTR) and Electronic Codebook (ECB) modes.

## Abstract
The research evaluates the effectiveness of various deep learning architectures, including:
- Convolutional Neural Networks (CNNs)
- Multi-Layer Perceptrons (MLPs)
- Recurrent Neural Networks (RNNs)

These models were trained to predict encryption keys based on known plaintext-ciphertext pairs. Despite achieving high training accuracy, the models struggled to generalize effectively to unseen data, indicating significant challenges in using AI for cryptanalysis.

## Key Findings
- RNNs showed better performance than CNNs in capturing cryptographic dependencies but suffered from vanishing gradients and model complexity issues.
- Models trained on small datasets performed well but failed to scale to larger keyspaces.
- The findings highlight the resilience of lightweight encryption algorithms like S-AES against deep learning-based attacks.

## Repository Structure
```
NU_Research/
├── data/               # Generated datasets used for training and testing
├── models/             # Implemented deep learning architectures
├── scripts/            # Python scripts for data generation, training, and evaluation
├── results/            # Experimental results and performance analysis
├── README.md           # Project documentation
```

## Installation
### Requirements
- Python 3.x
- TensorFlow/PyTorch
- NumPy
- Pandas
- Matplotlib

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Omar-Said-4/NU_Research.git
cd NU_Research
pip install -r requirements.txt
```

## Usage
### Data Generation
To generate encrypted datasets for S-AES:
```bash
python scripts/generate_data.py --mode CTR --size 100000
```

### Model Training
Train an RNN model on the dataset:
```bash
python scripts/train.py --model rnn --epochs 200
```

### Evaluation
Evaluate the trained model on the test dataset:
```bash
python scripts/evaluate.py --model rnn
```

## Citation
If you use this repository in your research, please cite our paper:
```
Said, O., Ismail, N., Yousef, S., Gamal, N. (2025). Assessing Deep Learning Challenges in Lightweight AES Cryptanalysis. In: Maglogiannis, I., Iliadis, L., Andreou, A., Papaleonidas, A. (eds) Artificial Intelligence Applications and Innovations. AIAI 2025. IFIP Advances in Information and Communication Technology, vol 756. Springer, Cham. https://doi.org/10.1007/978-3-031-96228-8_5
```

## Contributors
- **Sama Yousef** - [s-sama.yousef@zewailcity.edu.eg](mailto:s-sama.yousef@zewailcity.edu.eg)
- **Nada Ismail** - [s-nada.ismail@zewailcity.edu.eg](mailto:s-nada.ismail@zewailcity.edu.eg)
- **Omar Said** - [omar.Aziz02@eng-st.cu.edu.eg](mailto:omar.Aziz02@eng-st.cu.edu.eg)
- **Noha Gamal** - [NGamal@nu.edu.eg](mailto:NGamal@nu.edu.eg)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This research was supported by Nile University. We thank our advisors and colleagues for their valuable feedback and contributions.

