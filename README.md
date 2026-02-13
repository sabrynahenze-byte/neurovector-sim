# neurovector-sim
EEE 488 simulator to study device variability impacts on ANN/SNN performance.

The HTML page directory contains the code to run a classifier convieniently in a browser. There is a pretrained pytorch MNIST classification model (my_trained_mnist.pt). The script used to train it has also been included (train_model.py). 
Runnin app.py will start a server that will link index.html to port 5001, from there a user can load a jpeg and it will return classification likelihood. 
This is confirmed to run on macos and firefox. Doesn't appear to work on Chrome. Haven't tested on Windows.
Requirements.txt contains the package versions that this code used. 
