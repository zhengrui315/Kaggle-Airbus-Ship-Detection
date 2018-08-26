from _utils import *
import tensorflow as tf
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def create_submit():
    with tf.Session() as sess:
        model_dir = "./save_model"
        saver = tf.train.import_meta_graph(os.path.join(model_dir, "airbus_model.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        # if not in the original path, do:
        # saver.restore(sess, "new_path/airbus_model")

        graph = tf.get_default_graph()
        x_holder = graph.get_tensor_by_name("x_holder:0")
        pixel_pred = graph.get_tensor_by_name("pixel_pred:0")

        test_dir = "../data/test"
        image_id = np.array(os.listdir(test_dir))
        images_encode = {'ImageId':[], 'EncodedPixels':[]}
        for img_name in tqdm(image_id[:10]):
            image = cv2.imread(os.path.join(test_dir, img_name))/255.0
            # image = cv2.resize(image_shape)
            image = image[np.newaxis,:]
            pred = sess.run(pixel_pred, feed_dict={x_holder:image})
            masks = multi_rle_encode(pred)
            print(img_name)
            print(masks)
            if len(masks) > 0:
                for mask in masks:
                    images_encode['ImageId'].append(img_name)
                    images_encode['EncodedPixels'].append(mask)
            else:
                images_encode['ImageId'].append(img_name)
                images_encode['EncodedPixels'].append(None)

        submit_df = pd.DataFrame(images_encode)[['ImageId','EncodedPixels']]
        submit_df.to_csv('submission.csv', index=False)

        return submit_df


def check_submit(df):
    masks_agg['EncodedPixelsList'] = df.groupby('ImageId')['EncodedPixels'].apply(list)

    nrows = 2
    ncols = 5
    _, axes = plt.subplots(nrows, ncols, figsize=(24, 10))
    path = os.path.join(path, 'test')
    for idx in range(ncols):
        img_name = masks_agg.index[idx]
        mask = rle_decode_all(masks_agg.loc[img_name, 'EncodedPixelsList'])
        orig_img = plt.imread(os.path.join(path, img_name))

        axes[0, col].imshow(orig_img)
        axes[0, col].axis("off")
        axes[0, col].set_title(label='id ' + str(idx) + ':original', fontdict={'fontsize': 25})
        axes[1, col].imshow(mask * 200)
        axes[1, col].axis("off")
        axes[1, col].set_title(label='id ' + str(idx) + ': masks', fontdict={'fontsize': 25})
    plt.subplots_adjust(wspace=0.3, hspace=0.1)



if __name__ == '__main__':
    df = create_submit()
    #check_submit(df)
