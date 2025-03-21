# FashionVision-Hierarchical-CV-Blockchain

**FashionVision-Hierarchical-CV-Blockchain** is a deep learning project that combines CNN and a multi-stage hierarchical approach to classify fashion product images. The data is preprocessed (adding price, brand, etc.) and used to train a multi-task model. The trained model is then deployed via a Flask API for an e-commerce website that integrates blockchain technology – the blockchain part is not included in this repository. The API is used to automatically recognize product images and suggest product details to minimize manual data entry.

## Table of Contents

- [1. Background and Objectives](#1-background-and-objectives)
- [2. Project Structure](#2-project-structure)
- [3. System Requirements](#3-system-requirements)
- [4. Installation Guide](#4-installation-guide)
- [5. Key Files Description](#5-key-files-description)
- [6. How to Run the Training Notebooks](#6-how-to-run-the-training-notebooks)
- [7. How to Run the Flask API](#7-how-to-run-the-flask-api)
- [8. Extensions and Customizations](#8-extensions-and-customizations)
- [9. Contact Information](#9-contact-information)

---

## 1. Background and Objectives

- **Background:** In a complex fashion e-commerce system, manual product data entry can be time-consuming. With this project, sellers can simply upload a product image and the system automatically suggests:
  - Category: Shoes, shirts, bags, etc.
  - Key attributes: main color, gender, usage, brand, and more.
  - Suggested pricing, among others.
  
- **Objectives:**  
  1. Train a **CNN** model to recognize product information from images.  
  2. Build a multi-task model with a **multi-stage/hierarchical** approach (first predicting gender, main category, usage, then inferring detailed categories).  
  3. Package the model as a **Flask API** providing a `/predict` endpoint that returns classification results.

---

## 2. Project Structure

```
FashionVision-Hierarchical-CV-Blockchain/
├── CNN.ipynb
├── CNN_Multi_Stage.ipynb
├── flask_cnn.py
├── model.py
├── mappings.pkl
├── CNN_hierarchical_model.pt
├── requirements.txt
└── README.md
```

- **CNN.ipynb:** A notebook that demonstrates the CNN + MLP multi-task model.
- **CNN_Multi_Stage.ipynb:** A notebook with a multi-stage hierarchical model (with teacher forcing, transformer blocks, etc.).
- **flask_cnn.py:** A Flask API file that creates the `/predict` endpoint to process image URLs and return predictions.
- **model.py:** Contains the model class `EndToEndHierarchicalModel` (or your CNN/Multi-stage model classes).
- **mappings.pkl:** A pickle file that stores label-to-ID mappings (e.g., brand, gender, masterCategory, etc.).
- **CNN_hierarchical_model.pt:** The trained model’s `state_dict` file.
- **requirements.txt:** A list of required Python packages.
- **README.md:** This detailed project description and guide.

---

## 3. System Requirements

- **Python:** >= 3.8  
- **PyTorch:** (Install according to your system via instructions at [PyTorch.org](https://pytorch.org))  
- **Hardware:** CPU or GPU (GPU recommended for faster training)  
- **RAM:** Minimum 4GB (8GB+ recommended for full dataset training)

---

## 4. Installation Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<username>/FashionVision-Hierarchical-CV-Blockchain.git
   cd FashionVision-Hierarchical-CV-Blockchain
   ```
2. **Set up your virtual environment:**
   - Using **conda**:
     ```bash
     conda create -n fashionvision python=3.9
     conda activate fashionvision
     ```
   - Or using **venv**:
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Linux/Mac
     # or .\venv\Scripts\activate on Windows
     ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Key Files Description

### 5.1. `CNN.ipynb`
- Demonstrates a CNN + MLP multi-task model.
- Steps include:
  1. Loading the dataset from Hugging Face (`ashraq/fashion-product-images-small`).
  2. Preprocessing the data (adding price, brand, etc.).
  3. Splitting into train/validation/test sets.
  4. Building a model with ResNet (for image features) + MLP (for numerical features) that outputs 7 tasks (gender, masterCategory, usage, etc.).
  5. Training, evaluating, and testing the model.

### 5.2. `CNN_Multi_Stage.ipynb`
- Similar to the first notebook but implements a **multi-stage hierarchical model**:
  - **Stage 1:** Predict gender, masterCategory, and usage.
  - **Stage 2:** Uses embeddings from Stage 1 to predict subCategory, articleType, and baseColour.
  - **Stage 3:** Combines results from the previous stages to predict brand.
  - Implements **teacher forcing** with transformer blocks for improved accuracy.

### 5.3. `model.py`
- Contains the definition of the model class (e.g., `EndToEndHierarchicalModel`) used in the Flask API.
- You can separate the CNN + MLP classes if desired.

### 5.4. `flask_cnn.py`
- Implements a Flask API with a `/predict` route.
- Expects a JSON payload containing an `image_url`.
- Downloads the image, runs inference, and returns JSON with fields such as `brand, gender, usage, baseColour, masterCategory, subCategory, articleType, predicted_display_name`.

### 5.5. `mappings.pkl`
- Stores mappings between label names and IDs (e.g., brand, articleType, etc.) used for decoding model predictions.

### 5.6. `CNN_hierarchical_model.pt`
- Contains the trained model weights (saved as a `state_dict`), used for inference in `flask_cnn.py`.

---

## 6. How to Run the Training Notebooks

1. Open the notebook using Jupyter Notebook or VSCode.
2. Open either `CNN.ipynb` or `CNN_Multi_Stage.ipynb`.
3. Run the cells sequentially.
4. Monitor training/validation/test results.
5. Finally, save the trained model (`.pt` file) for inference.

> **Note:** To re-train the model, ensure you have an active internet connection for loading the dataset from Hugging Face and that all required packages are installed.

---

## 7. How to Run the Flask API

1. Ensure that `CNN_hierarchical_model.pt` and `mappings.pkl` are in the same directory.
2. Run the following command:
   ```bash
   python flask_cnn.py
   ```
3. The Flask server will start on port `5000` by default.
4. Send a `POST` request to `http://localhost:5000/predict` with a JSON payload like:
   ```json
   {
     "image_url": "https://link-to-your-image.jpg"
   }
   ```
5. The API will respond with a JSON object containing:
   ```json
   {
     "brand": "Nike",
     "gender": "Men",
     "usage": "Casual",
     "baseColour": "Black",
     "masterCategory": "Footwear",
     "subCategory": "Shoes",
     "articleType": "Sneakers",
     "predicted_display_name": "Nike Men Casual Black Footwear Shoes Sneakers"
   }
   ```

---

## 8. Extensions and Customizations

- **Stage Customization:**  
  You can modify the stages according to your business logic—add, remove, or rearrange stages as needed.

- **Output Adjustments:**  
  The current dataset outputs 7 labels; you can merge or split labels based on your requirements.

- **Deployment:**  
  - Use the Flask API in your blockchain-based e-commerce system.  
  - Consider Dockerizing the application for easier deployment.  
  - Deploy to cloud platforms such as AWS, GCP, etc.

---

## 9. Contact Information

- **Author:** ThanhVo - minthanh.codeai
- **Email:** thanhvo.contact@gmail.com
- **GitHub:** [ThanhVo15](https://github.com/ThanhVo15)

Thank you for your interest in the **FashionVision-Hierarchical-CV-Blockchain** project! If you have any questions or feedback, please open an issue or contact me directly.



