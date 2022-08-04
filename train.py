import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil
from tensorboardX import SummaryWriter
from evaluate import evaluate_dataset, classification_accuracy
from models import *
from MNISTDataset import MNISTDataset
import utils


def train_model(model, n_epochs, batch_size, learning_rate):
    # Create the model
    if model == 'linear':
        model_name = model
        model = LinearKernelCNN(in_channels=3, n_classes=10)
    elif 'mlp' in model:
        model_name = model
        h = int(model.split('_')[1].split('=')[1])
        model = MLPKernelCNN(in_channels=3, n_classes=10, n_hidden=int(h))
    else:
        raise NotImplementedError()

    # Send the model to the GPU if one is available
    if torch.cuda.is_available():
        model.cuda()

    # Create the optimizer. Adam is a popular improvement on vanilla stochastic gradient descent.
    # You just need to give it the parameters of the model and the learning rate at which you want to train.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate the dataloaders that well feed you batches of the inputs+labels in your dataset
    train_set = MNISTDataset('data/train', resolution=[32, 32], training=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    test_set = MNISTDataset('data/test', resolution=[32, 32])
    test_loader = DataLoader(test_set, num_workers=2)

    # Prepare save directories and configure tensorboard logging
    shutil.rmtree(os.path.join('saved_runs', model_name), ignore_errors=True)   # Delete the existing save folder
    os.mkdir(os.path.join('saved_runs', model_name))                            # Create the save folder
    writer = SummaryWriter(os.path.join('saved_runs', model_name, 'logs'))      # Specify where to save training logs

    print_freq = 50                             # How often to print training metrics in the console
    running_metrics = None                      # Keep running average of training metrics (e.g. loss) over each epoch
    for epoch in range(1, n_epochs + 1):
        print('Starting epoch %d\n' % epoch)

        for batch, data in enumerate(train_loader):                 # Index of current batch and inputs+labels
            step_num = utils.step_num(epoch, batch, train_loader)   # Global training step (number of iterations up until now)

            # Obtain the inputs/targets
            images, labels = data
            labels = labels.view(-1)    # Labels must be 1 dimensional with class indices, but DataLoader creates batch dimension
            # Send to GPU if possible
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            # Compute the losses
            predictions = model(images)                     # Get the model output
            loss = F.cross_entropy(predictions, labels)     # Compute the loss between model output and true labels

            # Backprop and optimize
            optimizer.zero_grad()   # Zero out the current .grad properties on all model parameters
            loss.backward()         # Backpropagate the loss and populate the .grad properties
            optimizer.step()        # Optimizer uses the .grad values to modify the parameters by taking a step down the gradient

            # Compute error metrics and update running losses
            accuracy = classification_accuracy(predictions, labels)
            losses = {'loss': loss.item(), 'accuracy': accuracy}    # .item() obtains the raw value of a 0D Tensor
            running_metrics = utils.update_metrics(running_metrics, losses, print_freq)    # Running metrics since last log

            # Print metrics to the console
            if (step_num + 1) % print_freq == 0:
                utils.log_to_tensorboard(writer, running_metrics, step_num)      # Log metrics to tensorboard for later visualization
                utils.print_metrics(running_metrics, step_num, n_epochs * len(train_loader))  # Print metrics to the console
                running_metrics = {l: 0 for l in running_metrics}                 # Reset the running losses

        print('Finished epoch %d\n' % epoch)

        print('Evaluation')
        eval_metrics = evaluate_dataset(model, test_loader)
        utils.log_to_tensorboard(writer, eval_metrics, step_num, training=False)
        utils.print_metrics(eval_metrics, step_num, n_epochs * len(train_loader))

        utils.save_model(model, os.path.join('saved_runs', model_name))


if __name__ == '__main__':
    # Training configuration
    model = 'mlp_h=3'
    n_epochs = 2
    batch_size = 16
    learning_rate = 1e-3

    train_model(model, n_epochs, batch_size, learning_rate)
