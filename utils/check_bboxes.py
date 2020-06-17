from dataset import *


def main():
    print('train fold')
    check_fold('train')
    print('val fold')
    check_fold('val')


def check_fold(fold_name):
    dataset = WheatDataset(
        f'./data/train', f'./data/{fold_name}.csv', transforms=None)

    for i in range(len(dataset)):
        _, bboxes, _ = dataset[i]
        for bbox in bboxes:
            if bbox[0] + bbox[2] > 1024.0 or bbox[1] + bbox[3] > 1024.0:
                print(f'{i}: {bbox}')


if __name__ == "__main__":
    main()
