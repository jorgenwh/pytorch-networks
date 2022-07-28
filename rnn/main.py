import torch
import numpy as np
from keras.datasets import mnist

from rnn import RNNModel
from utils import one_hot_encode, get_tqdm_bar, AverageMeter

INPUT_SIZE = 28
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 10
NUM_LAYERS = 1
SEQ_LENGHT = 28

CUDA = True
EPOCHS = 14
LEARNING_RATE = 0.2
BATCH_SIZE = 8

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = one_hot_encode(train_labels)

    num_training_examples = train_images.shape[0]
    num_testing_examples = test_images.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")

    model = RNNModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            num_layers=NUM_LAYERS,
            seq_length=SEQ_LENGHT
    ).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    print("Training model")
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        epoch_loss = AverageMeter()

        bar = get_tqdm_bar(range(0, num_training_examples, BATCH_SIZE), "Training")
        for i in bar:
            x = train_images[i:i+BATCH_SIZE]
            t = train_labels[i:i+BATCH_SIZE]

            x = torch.Tensor(x).to(device)
            t = torch.Tensor(t).to(device)

            y, hidden = model.forward(x)
            y = y[:,-1,:] # We only use the final sequence output

            loss = criterion(y, t)

            epoch_loss.update(loss.item(), BATCH_SIZE)
            bar.unit = f"Loss: {epoch_loss}"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Inference
    print("\nPerforming inference")
    with torch.no_grad():
        bar = get_tqdm_bar(range(num_testing_examples), "Inference")
        correct = 0
        inferred = 0

        for i in bar:
            x = test_images[i]
            t = test_labels[i]

            x = torch.Tensor(x).to(device)

            y, hidden = model.forward(x)
            y = torch.softmax(y[-1], dim=0)
            y = np.argmax(y.cpu().detach().numpy())

            inferred += 1
            correct += int(y == t)
            bar.unit = f"Accuracy: {round(correct / inferred, 3)}"

    print(f"Inference accuracy: {round(correct / inferred, 3)}")
