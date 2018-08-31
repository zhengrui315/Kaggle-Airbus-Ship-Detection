from model import airbus_vgg, airbus_scratch
from _utils import *
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():

    parser = argparse.ArgumentParser(description='Airbus Nails it')
    parser.add_argument(
        '-n',
        '--num_epochs',
        type=int,
        nargs='?',
        default=1,
        help='Number of epochs.'
    )
    parser.add_argument("-ct", "--continues_training", help="Continue from where you left off", action="store_true")
    args = parser.parse_args()
    NUM_EPOCHS = args.num_epochs
    CONTINUE_TRAINING = args.continues_training

    #image_shape = (224, 224)
    data_folder = os.path.join(os.getcwd(), '../data/train')
    model_folder = os.path.join(os.getcwd(), 'save_model')
    label_df = masks_read(os.path.join(os.getcwd(), '../data'))

    airbus_model = airbus_vgg(model_dir=model_folder, continue_training=CONTINUE_TRAINING)
    #airbus_model.debug2(data_folder, label_df)
    airbus_model.train(NUM_EPOCHS, data_folder, label_df)


if __name__ == '__main__':
    main()