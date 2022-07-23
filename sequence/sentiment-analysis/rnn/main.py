import torch

from utils import AverageMeter, get_vocab, generate_training_data
from rnn import RNN 

if __name__ == "__main__":
    sentences = [
            "i like this film", 
            "this film is not bad", 
            "i dislike this film", 
            "this film is bad", 
            "i liked parts of the introduction but hated the film overall", 
            "awfully good film", 
            "awfully bad film"
    ]
    labels = [1, 1, 0, 0, 0, 1, 0]

    vocab = get_vocab(sentences)

    x_train, y_train = generate_training_data(sentences, labels, vocab)

    cuda = False
    epochs = 10
    learning_rate = 0.01

    input_size = len(vocab)
    hidden_size = 25
    output_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}\n")
    model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # Binary Cross Entropy Loss
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        epoch_loss = AverageMeter()

        for x, t in zip(x_train, y_train):
            y = model.forward(x)

            loss = criterion(y, t)

            epoch_loss.update(loss.item())
            print(f"Loss: {epoch_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Check the predictions to see whether it has learned the training data
    for i, (x, t) in enumerate(zip(x_train, y_train)):
        y = model.forward(x)

        print(f"\nsentence: '{sentences[i]}'")
        print(f"output: {y}")
        print(f"label: {t}")
