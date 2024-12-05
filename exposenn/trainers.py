import torch
import torch.nn.functional as F
from exposenn.loss import SemanticLoss


def train(model, dataloader, loss_fn, optimizer, *,
          max_epochs=None, val_dataloader=None, eval_epochs=1,
          callbacks=None):
    """
    Trains a PyTorch model with the given data and optimization routine, optionally evaluating on a validation set
    and calling custom callbacks at each epoch.

    The function handles model training over multiple epochs, computes training and validation losses, and allows
    for flexible integration of loss functions, including those that might involve additional concepts.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained. The model should return both main task predictions (y_hat) and concept
        predictions (concepts_hat) when given input data.
    dataloader : torch.utils.data.DataLoader
        The dataloader for the training dataset. It should yield batches of (X, concepts, y), where:
        - X: The input data.
        - concepts: The ground truth concept values (if applicable).
        - y: The ground truth target labels.
    loss_fn
        The loss function used for training. This function should take two arguments: the predictions (both main
        task and concept outputs) and the ground truth (both target labels and concepts).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's parameters.
    max_epochs : int, optional
        The maximum number of epochs to train the model. If `None`, the training loop will continue indefinitely
        until manually stopped.
    val_dataloader : torch.utils.data.DataLoader, optional
        The dataloader for the validation dataset. If provided, the model will be evaluated on the validation set
        every `eval_epochs` epochs.
    eval_epochs : int, optional
        The frequency (in epochs) at which the model should be evaluated on the validation set. Default is 1 (i.e.,
        evaluate at the end of every epoch).
    callbacks : list of callable, optional
        A list of callback functions to be called at the end of each epoch. Each callback should take two arguments:
        the current epoch number and a dictionary containing the training and validation loss for that epoch.

    Returns
    -------
    history : list[dict]
        A list of dictionaries where each dictionary contains the training and validation loss for each epoch.
    """
    
    if callbacks is None:
        callbacks = []
    history = []
    epoch = 0
    while True:
        history.append({})
        model.train()
        for X, concepts, y in dataloader:
            optimizer.zero_grad()
            y_hat, concepts_hat = model(X)

            if isinstance(loss_fn.concepts_loss_fn, SemanticLoss):
                concepts_values = (concepts_hat, F.sigmoid(y_hat))
            else:
                concepts_values = (concepts_hat, concepts.to(torch.float))

            loss = loss_fn(concepts_values,
                           (y_hat, y.to(torch.float)))
            loss.backward()
            optimizer.step()
        history[-1]['train_loss'] = loss.detach().item()
        
        if val_dataloader is not None and epoch % eval_epochs == 0:
            model.eval()
            y_hat, concepts_hat = model(X)

            if isinstance(loss_fn.concepts_loss_fn, SemanticLoss):
                concepts_values = (concepts_hat, F.sigmoid(y_hat))
            else:
                concepts_values = (concepts_hat, concepts.to(torch.float))

            val_loss = loss_fn(concepts_values,
                               (y_hat, y.to(torch.float)))
            history[-1]['val_loss'] = val_loss.detach().item()
        
        for callback in callbacks:
            callback(epoch, history[-1])
            
        epoch += 1
        if max_epochs is not None and epoch >= max_epochs:
            return history
