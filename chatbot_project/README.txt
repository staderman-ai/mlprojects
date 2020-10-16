This project reads training data from a json file. The training data is divided according to topics. For each topic there are a set of questions and answers. 

The training data is modified into a set of bow (bag of words) and the corresponding topic name. The data is one-hot encoded. The neural network consists of 2 Dense layers followed by a Dense layer having size = number of topics. This NN is fitted onto the training data. When a query is raised in the chatbot window, it is processed and converted into a bow (bag of words), fed into the NN and the NN response gives the topic of discussion. This is then used to randomly select from one of several answers corresponding to that particular topic.

Note that the chatbot UI code has been borrowed and integrated.
