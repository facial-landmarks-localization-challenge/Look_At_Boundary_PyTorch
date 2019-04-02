import torch.utils.data as data
from utils import get_annotations_list, getitem_from


class GeneralDataset(data.Dataset):

    def __init__(self, dataset='WFLW', split='train', eval_flag=0):
        self.dataset = dataset
        self.split = split
        self.eval_flag = eval_flag
        self.list = get_annotations_list(dataset, split, ispdb=abs(1-eval_flag))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        return getitem_from(self.dataset, self.split, self.list[item], self.eval_flag)


# TEST: dataset
if __name__ == '__main__':
    import cv2
    from utils import dataset_kp_num
    use_set = '300W'
    datasets = GeneralDataset(use_set, 'train')
    dataloader = data.DataLoader(datasets, batch_size=1, shuffle=False, pin_memory=True)
    print('Number of training batches per epoch: %d' % len(dataloader))
    for ddd, context in enumerate(dataloader):
        image, keypoint, _ = context
        image = image.squeeze().numpy()
        keypoint = keypoint.squeeze().numpy()
        for coord_index in range(dataset_kp_num[use_set]):
            cv2.circle(image, (int(keypoint[2 * coord_index]), int(keypoint[2 * coord_index + 1])), 1, (0, 0, 255))
        cv2.imshow('pic', image)
        cv2.moveWindow('pic', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ddd > 10:
            break
