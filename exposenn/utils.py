import torch.nn as nn

from exposenn import models


def find_layer_predicate_recursive(model, predicate):
    """
    Recursively searches through a PyTorch model and returns a list of all layers that satisfy a given predicate.

    This function explores the model's architecture, including all submodules, and applies the provided
    predicate function to each layer. If the predicate returns `True` for a given layer, the layer is added
    to the result list.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to search through. The model can contain nested submodules, and the function will
        recursively traverse all levels of the model hierarchy.
    predicate : function
        A function that takes a PyTorch layer (an instance of `torch.nn.Module`) as input and returns a
        boolean (`True` or `False`). Layers for which the predicate returns `True` are included in the output list.

    Returns
    -------
    list
        A list of layers from the model that satisfy the given predicate function. Each item in the list
        is a layer that passed the condition defined by the predicate.
    """

    layers = []
    for name, layer in model._modules.items():
        if predicate(layer):
            layers.append(layer)
        layers.extend(find_layer_predicate_recursive(layer, predicate))
    return layers


def find_layers_types_recursive(model, layer_type_list):
    """
    Recursively searches through a PyTorch model and returns a list of all layers that match one or more
    specified types.

    This function traverses the entire structure of the given model, including nested submodules (e.g.,
    layers inside of `torch.nn.Sequential` or custom `torch.nn.Module` compositions), and collects
    all layers that match any of the types listed in `layer_type_list`.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to search through. This can be any custom or predefined model, including those
        with nested layers and submodules.
    layer_type_list : list
        A list of PyTorch layer types (e.g., `torch.nn.Conv2d`, `torch.nn.Linear`, etc.) to search for within
        the model. The function will return all layers that are instances of any of these types.

    Returns
    -------
    list
        A list of all layers in the model that are instances of any of the given types. Each item in the list
        is a layer object from the model that matches one of the types in `layer_type_list`.
    """

    def predicate(layer):
        return type(layer) in layer_type_list

    return find_layer_predicate_recursive(model, predicate)


def get_output_features(layer):
    """
    Returns the number of output features (or channels) from the given layer of a PyTorch model.
    This function is designed to handle various layer types commonly used in neural networks, such as
    convolutional, fully connected, and recurrent layers.

    Parameters
    ----------
    layer
        A PyTorch layer (defined in the torch.nn package) from which the number of output features or
        channels will be extracted. This can be a convolutional layer (e.g., `nn.Conv2d`), a fully connected
        layer (e.g., `nn.Linear`), or other supported types.

    Returns
    -------
    int
        The number of output features (or channels) of the layer. Depending on the layer type, this could
        represent the number of neurons in a fully connected layer (`out_features`), the number of output
        channels in a convolutional layer (`out_channels`), or other similar attributes.

    Notes
    -----
    This function relies on specific attributes being present in the layer:
    - `out_features` (e.g., in `nn.Linear` layers),
    - `num_features` (e.g., in `nn.BatchNorm2d` layers),
    - `out_channels` (e.g., in `nn.Conv2d` layers),
    - `hidden_size` (e.g., in recurrent layers like `nn.LSTM` or `nn.GRU`).
    """

    if hasattr(layer, 'out_features'):
        return layer.out_features
    elif hasattr(layer, 'num_features'):
        return layer.num_features
    elif hasattr(layer, 'out_channels'):
        return layer.out_channels
    elif hasattr(layer, 'hidden_size'):
        return layer.hidden_size
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")


def create_connectors(backbone, layer_type_list):
    """
    Creates connectors to link specific layers of a base network (backbone) to an interpretation network.
    The connectors process the output of the selected layers in the base network and form a tensor that
    is then passed as input to the interpretation network.

    Parameters
    ----------
    backbone : torch.nn.Module
        The base network, for whose layer outputs you need to create connectors. This network can include various types
        of layers such as convolutional, fully connected, normalization layers.
    layer_type_list : list
        A list of layer types (e.g., `torch.nn.Conv2d`, `torch.nn.Linear`, etc.) to look for in the base network.
        The function will create connectors for layers of these specific types.

    Returns
    -------
    connectors : list[tuple]
        A list of tuples (layer, connector), where `layer` is a layer from the base network, and `connector` is
        a `torch.nn.Module` that processes the output of this layer and prepares it to be passed into the interpretation network.
    total_features : int
        The total number of output features (channels) from all the base network layers for which connectors were created.
        This value is used to configure the interpretation network that receives these outputs.

    Raises
    ------
    ValueError
        If an unsupported layer type is encountered during connector creation.

    Example
    -------
    model = BaseModel()
    layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    connectors, total_features = create_connectors(model, layer_types)
    """

    layers = find_layers_types_recursive(backbone, layer_type_list)

    total_features = 0
    connectors = []

    for index, layer in enumerate(layers):
        num_features = get_output_features(layer)

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
            connector = models.GlobalAvgPool2dConnector(num_features)
        elif isinstance(layer, nn.Linear):
            connector = nn.Identity()
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        total_features += num_features
        connectors.append((layer, connector))

    return connectors, total_features
