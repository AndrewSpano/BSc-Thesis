import torch

from transformers import RobertaTokenizerFast


def mlm(
        batch_input_ids: torch.Tensor,
        tokenizer: RobertaTokenizerFast,
        mask_probability_p: float = 0.15
) -> torch.Tensor:
    """Randomly masks the parts of the given tensor, according to a masking
        probability `p`."""
    # clone the array and fix its shape to [B, maxlen]
    batch_input_ids_ = batch_input_ids.detach().clone()
    if len(batch_input_ids_.shape) == 1:
        batch_input_ids_ = batch_input_ids_.reshape(1, -1)

    # random floats in [0, 1] for each token, for each batch
    random_array = torch.rand(batch_input_ids_.shape)

    # mask tokens that are not [<s>, </s>, <pad>] according to the
    #  masking probability
    valid_indices = (batch_input_ids_ != tokenizer.bos_token_id) * \
                    (batch_input_ids_ != tokenizer.eos_token_id) * \
                    (batch_input_ids_ != tokenizer.pad_token_id)
    masked_array = (random_array < mask_probability_p) * valid_indices

    # for each batch element
    for i in range(batch_input_ids_.shape[0]):
        # get the indices of the tokens that will be masked, as a flat list
        mask_indices = torch.flatten(masked_array[i].nonzero()).tolist()
        # mask the corresponding indices for this batch
        batch_input_ids_[i, mask_indices] = tokenizer.mask_token_id

    return batch_input_ids_
