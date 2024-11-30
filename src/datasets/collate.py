import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["audio"] = torch.nn.utils.rnn.pad_sequence(
        [elem["audio"].squeeze() for elem in dataset_items],
        batch_first=True,
    ).unsqueeze(1)
    result_batch["wav_path"] = [elem["wav_path"] for elem in dataset_items]
    result_batch["sample_rate"] = dataset_items[0]["sample_rate"]

    return result_batch
