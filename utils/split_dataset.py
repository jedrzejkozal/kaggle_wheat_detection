import csv
import sklearn.model_selection
import collections
import numpy as np


def main():
    x_all = []
    y_all = []

    all_data = collections.defaultdict(list)
    class_counts = collections.defaultdict(int)

    with open('./data/all.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == 'image_id':
                continue
            filename, width, height, bbox, class_name = row
            if len(x_all) == 0 or x_all[-1] != filename:
                x_all.append(filename)
                y_all.append(class_name)
            all_data[filename].append(row)
            class_counts[class_name] += 1
    print(class_counts)

    print('x_all = ', len(x_all))
    print('y_all = ', len(y_all))

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_all, y_all, test_size=0.3, stratify=y_all)

    print('x_train = ', len(x_train))
    print('y_train = ', len(y_train))
    print('x_val = ', len(x_val))
    print('y_val = ', len(y_val))

    with open('./data/train.csv', 'w+') as f:
        writer = csv.writer(f)
        for filename in x_train:
            for row in all_data[filename]:
                writer.writerow(row)

    with open('./data/val.csv', 'w+') as f:
        writer = csv.writer(f)
        for filename in x_val:
            for row in all_data[filename]:
                writer.writerow(row)


if __name__ == '__main__':
    main()
