import csv


def main():
    filename_to_class = {}
    with open('./data/all.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == 'image_id':
                continue
            filename, _, _, _, class_name = row
            if filename in filename_to_class:
                if class_name != filename_to_class[filename]:
                    raise RuntimeError("Not all images have one label")
            else:
                filename_to_class[filename] = class_name

        print("All images have one label")


if __name__ == '__main__':
    main()
