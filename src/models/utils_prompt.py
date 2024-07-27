import torch

def init_prompt(x, prompt_pos):
    if x.shape[1] == 197:
        pass
    else:
        x = torch.cat((
                x[:, :1, :],
                x[:, (1+ prompt_pos):, :]
            ), dim=1)
    return x


def update_prompt(x, prompt, dataset, layer_index):
    if x.shape[1] == 197:
        x = torch.cat((
                x[:, :1, :],
                prompt,
                x[:, 1:, :]
            ), dim=1)
    else:
        if not (layer_index == 11 and dataset == 'imagenet_r'):
            x = torch.cat((
                    x[:, :1, :],
                    prompt,
                    x[:, 1:, :]
                    ), dim=1)
    return x