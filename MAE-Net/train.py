import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import skimage.io as sio
import numpy as np
import tensorflow as tf
import tqdm

import mae_net
import data
import utils
import checkpoints

ap = argparse.ArgumentParser()

ap.add_argument('--datasets_dir', default='dataset/')
ap.add_argument('--load_size', type=int, default=286)  # Load the images with this size.
ap.add_argument('--crop_size', type=int, default=256)  # Cropping to this size.
ap.add_argument('--batch_size', type=int, default=1)
ap.add_argument('--n', type=int, default=3)  # Order of the operational layer (n parameter).
# epoch default = 2000
ap.add_argument('--epochs', type=int, default=2000)
ap.add_argument('--epoch_decay', type=int, default=100)  # After this epoch, start learning rate decay.
ap.add_argument('--lr', type=float, default=0.0002)
ap.add_argument('--beta_1', type=float, default=0.5)
ap.add_argument('--cycle_loss_weight', type=float, default=10.0)
ap.add_argument('--identity_loss_weight', type=float, default=5.0)
ap.add_argument('--class_loss_weight', type=float, default=0.1)
ap.add_argument('--pool_size', type=int, default=50)  # Pool size for storing fake samples.
ap.add_argument('--method', help='operational, convolutional, convolutional-light', default='operational')
args = vars(ap.parse_args())

if not os.path.exists('output'): os.makedirs('output')

# Loading data.
A_img_paths, A_label = utils.readData(args['datasets_dir'] + 'trainA' + '/*.png')
B_img_paths, B_label = utils.readData(args['datasets_dir'] + 'trainB' + '/*.png')

A_img_paths_test, A_label_test = utils.readData(args['datasets_dir'] + 'testA' + '/*.png')
B_img_paths_test, B_label_test = utils.readData(args['datasets_dir'] + 'testA' + '/*.png')

A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths,
                                                 A_label, B_label,
                                                 args['batch_size'], args['load_size'], args['crop_size'],
                                                 training=True, repeat=False)

A2B_pool = data.ItemPool(args['pool_size'])
B2A_pool = data.ItemPool(args['pool_size'])

A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test,
                                            A_label_test, B_label_test,
                                            args['batch_size'], args['load_size'], args['crop_size'],
                                            training=False, repeat=True)

# Creating models.
mae_net = mae_net.mae_net()
mae_net.init(args, len_dataset)

# 可视化网网络结构
tf.keras.utils.plot_model(mae_net.G_A2B, to_file="SNP_G_A2B.png", show_shapes=True)
tf.keras.utils.plot_model(mae_net.G_B2A, to_file="SNP_G_B2A.png", show_shapes=True)
tf.keras.utils.plot_model(mae_net.D_A, to_file="SNP_D_A.png", show_shapes=True)
tf.keras.utils.plot_model(mae_net.D_B, to_file="SNP_D_B.png", show_shapes=True)


def train_step(A, B):
    A2B, B2A = mae_net.train_G(A, B)
    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    mae_net.train_D(A[0], B[0], A2B, B2A)


# Initialize the epoch counter.
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# Create the checkpoint.
checkDir = 'output/checkpoints/' + args['method']
if not os.path.exists(checkDir): os.makedirs(checkDir)
checkpoint = checkpoints.Checkpoint(dict(G_A2B=mae_net.G_A2B,
                                         G_B2A=mae_net.G_B2A,
                                         D_A=mae_net.D_A,
                                         D_B=mae_net.D_B,
                                         G_optimizer=mae_net.G_optimizer,
                                         D_optimizer=mae_net.D_optimizer,
                                         ep_cnt=ep_cnt),
                                    checkDir,
                                    max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
    print('\nCheckpoint is restored!\n')
except Exception as e:
    print(e)

test_iter = iter(A_B_dataset_test)

sample_dir = 'output/samples_training/'
if not os.path.exists(sample_dir): os.makedirs(sample_dir)

# Main loop for the training.
for ep in tqdm.trange(0, args['epochs'] + 1, desc='Epoch Loop'):
    if ep < ep_cnt:
        continue

    # Update the epoch counter.
    ep_cnt.assign_add(1)

    # Train for an epoch.
    for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
        train_step(A, B)

        # Samples for the restoration during the training.
        if mae_net.G_optimizer.iterations.numpy() % 500 == 0:
            A, B = next(test_iter)
            A2B, B2A, A2B2A, B2A2B = mae_net.sample(A[0], B[0])
            img = utils.immerge(np.concatenate([A[0], A2B, A2B2A, B[0], B2A, B2A2B], axis=0), n_rows=2)
            sio.imsave(sample_dir + 'iter-' + str(mae_net.G_optimizer.iterations.numpy()) + '.jpg',
                       ((img + 1.) / 2. * 255).astype(np.uint8), quality=95)

    # Save the current state.
    checkpoint.save(ep)
