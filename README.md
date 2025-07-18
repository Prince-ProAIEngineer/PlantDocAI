# PlantDocAI
Deep Learning-Based Plant Disease Detection with ResNet-18 using Pytorch
This project uses a fine-tuned ResNet-18 model to detect plant leaf diseases from images. Trained on labeled plant disease datasets, the model achieves high classification accuracy(96.43%) and offers real-world application for early plant disease diagnosis.
This model has been trained to detect only 35+ diseases for now. But this model can be trained for even more disease classification in more plants. This shows an idea with a huge potential to transform the agricultural dynamics in rural area whuch lack expert agricultutist for plant disease detection.
This model is built for speed, accuracy, and deployment-readiness, this solution combines a robust training pipeline in Jupyter Notebook with a minimal standalone inference script.

Objectives
->Detect diseases in plant leaves with high accuracy using deep learning.

->Train and evaluate using a reproducible PyTorch pipeline.

->Deploy lightweight inference with a single image input.

->Make agricultural diagnostics accessible and scalable.

Model Architecture
->Base Model: ResNet-18 (pretrained on ImageNet)

->Final Layer: Custom linear layer matching the number of disease classes

->Optimizer: AdamW

->Loss Function: CrossEntropyLoss

->Input Size: 224×224 RGB image

Dataset:
All images resized to 224×224 during processing
Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Accuracy: 96.43%

Future Work
->Add Grad-CAM explainability

->Deploy with FastAPI or Streamlit

->Extend to mobile deployment with ONNX

Author
Prince Kumar Mandal
Gold Medalist - National AI Olympiad 🇳🇵

Acknowledgments
->PyTorch Team

->New Plant Diseases Dataset

->ResNet authors

If you find this useful, star the repo to support the project!
