def get_flat_number(tensor=None, sizes=None, start_dim=1):
    """
    :param tensor: the tensor to get flat size from
    :param start_dim: which dim to flatten from (defaulted to 1 so as to not flatten batch dim) set to 0 to flatten whole tensor
    :return: the length of the tensor if it were flattened to a single dimension (from start dim)
    """
    prod = 1
    if not (tensor is None):
        items = list(tensor.size())
    elif not (sizes is None):
        items = sizes
    else:
        raise Exception("Error using get flat number - both tensor and sizes arg is none")
    for i in range(start_dim, len(items)):
        prod *= items[i]

    return prod
