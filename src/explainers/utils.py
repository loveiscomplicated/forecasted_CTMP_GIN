import itertools

def _iter_selected_batches(dataloader, selected_indices):
    """
    Yield only selected batches from dataloader, without iterating the full loader.

    Args:
        dataloader: PyTorch DataLoader
        selected_indices: sorted list of batch indices to keep

    Yields:
        (batch_index, batch)
    """
    it = iter(dataloader)
    prev = -1
    for idx in selected_indices:
        skip = idx - prev - 1
        if skip > 0:
            it = itertools.islice(it, skip, None)
        batch = next(it)
        yield idx, batch
        prev = idx