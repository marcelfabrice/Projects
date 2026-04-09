from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, r2_score
import matplotlib.pyplot as plt
import numpy as np

def test(nn, X_test, y_test, criterion, classification, visualize=False):
    test_losses, predictions = [], []
    for X, y_true in zip(X_test, y_test):

        #Ausgabe vom Netz
        output ,_  = nn(X) 
        
        #Onehot bei klassifikation
        if classification:
            y_pred = np.argmax(output)
        else: 
            y_pred = float(output)

        #Loss Function des Netzes
        criterion.y_true = y_true
        criterion.y_pred = y_pred

        #für plots
        predictions.append(y_pred)
        test_losses.append(criterion.value())

    precision = precision_score(y_test, predictions, average='macro')
    accuracy = accuracy_score(y_test, predictions)

    if visualize:
        precision = precision_score(y_test, predictions, average='macro')
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()

    return precision, accuracy

def test_samples(NeuralNetwork, data):
    fig = plt.figure(figsize=(10,5))
    for i in range(1, 11):
        ax = fig.add_subplot(2,5,i)
        index = np.random.randint(0, len(data.data))
        image = data.images[index]
        label = data.target[index]
        x = data.data.reshape(-1, 8, 8, 1).astype(np.float32)
        output,_ = NeuralNetwork(x[index])
        predicted_class = int(np.argmax(output))
        ax.imshow(image, cmap='gray')
        ax.set_title(f"true: {label} pred: {predicted_class}")
        ax.axis('off')
    plt.show()