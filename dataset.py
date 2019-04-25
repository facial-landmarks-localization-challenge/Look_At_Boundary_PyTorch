import torch.utils.data as data
from utils import args, get_annotations_list, get_item_from


class GeneralDataset(data.Dataset):

    def __init__(self, dataset='WFLW', split='train'):
        self.dataset = dataset
        self.split = split
        self.list = get_annotations_list(dataset, split, ispdb=args.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        return get_item_from(self.dataset, self.split, self.list[item])
