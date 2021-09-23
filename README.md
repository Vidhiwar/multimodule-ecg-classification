# multimodule-ecg-classification

RESEARCH-PAPER:{

      TITLE: "Multi-module Recurrent Convolutional Neural Network with Transformer Encoder for ECG Arrhythmia Classification",

      CITE:  "https://ieeexplore.ieee.org/abstract/document/9508527",
      
      YEAR: 2021,
      
      CONFERENCE: "IEEE EMBS",
      
      AUTHORS: ["Duc Le^",
                    "Vidhiwar Singh Rathour^",
                          "Sang Truong", 
                                   "Quan Mai^, 
                                            Patel Brijesh; 
                                                    Ngan Le"],
                                                          ^: Equal Contribution}  

                                                          
                                                                                    
DIRECTORY-TREE:{

     data: "Directory: Datasets for training are stored here.",
    
     utils: "Directory: Utility based files",
     
     examples: {"Directory: github/awni/ecg/":[
            "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network"]},

     models:{ "Directory: DNN Models": {
            resnet_cnn.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, CNN, Word2Vec",
            resnet_lstm_phy2017.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN",
            resnet_lstm.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN",
            resnet_w2v.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN, Word2Vec",
            resnet_lstm_mitbih.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN, Word2Vec, Attention AYN"}},

     ecg_cnn.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, CNN, Resnet",
     ecg_w2v.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, Word2Vec, Resnet",
     ecg_lstm.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN, LSTM",
     ecg_phy2017.py:"Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN",
     ecg_mitbih.py: "Python: Pytorch implememntation of Multimodal-ecg-classification, LSTM, CNN, Word2Vec, Attention AYN",
     transform_data.ipynb: "Jupyter Notebook: Python implementation for Data Generation, and Preprocessing"}
                                       
HOW-TO-USE:{

      Uno: "Make sure the required libraries (Torch, Panda, Tqdm, ... etc.,). are installed",
      Dos: : "Use the examples directory to download and preprocess data.",
      Tres: "Follow transform_data.ipyn to get data ready for training.",
      Cuatro: "Run python ecg_###.py to train on training data, and validate on validation data",
      Cinco: "By default results are saved in checkpoints directory"}
                               
IMAGES:{ 

 [![Model.png](https://i.postimg.cc/PJw2b5gd/Model.png)](https://postimg.cc/vxGrbb8K)
 [![Results.png](https://i.postimg.cc/PxmCmGSB/Results.png)](https://postimg.cc/34xrTq5B)}

             
#  EOF
                     
                    
