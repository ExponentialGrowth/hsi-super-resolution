from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['/Users/apple/Downloads/train2014',
                                     '/Users/apple/Downloads/val2014'],
                      test_folders=['/Users/apple/Documents/GitHub/a-PyTorch-Tutorial-to-Super-Resolution/test_dataset'],
                      min_size=100,
                      output_folder='./')
