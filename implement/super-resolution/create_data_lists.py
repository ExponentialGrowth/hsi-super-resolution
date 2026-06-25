from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['/Users/apple/Downloads/train2014',
                                     '/Users/apple/Downloads/val2014'],
                      test_folders=["/Users/apple/Downloads/BSDS100/original",
                                    "/Users/apple/Downloads/Set5/original",
                                    "/Users/apple/Downloads/Set14/original"],
                      min_size=100,
                      output_folder='./')
