import tensorflow as tf
import numpy as np


class Dataset(object):
    def __init__(self, configs, current_directory, buffer_size_param=50, num_parallel_calls=6):
        self.crop_size_h = configs.crop_size[0]
        self.crop_size_w = configs.crop_size[1]
        self.batch_size = configs.batch_size
        self.batch_size_test = configs.batch_size if configs.AE_only else 1  # when training with SI (batch size=1)
        self.iterations = configs.iterations
        self.root_data = configs.root_data
        self.num_crops_per_img = configs.num_crops_per_img  # number of patches to crop from img
        self.do_flips = configs.do_flips
        self.buffer_size = buffer_size_param * self.num_crops_per_img
        self.num_parallel_calls = num_parallel_calls
        self.file_path_train = current_directory + configs.file_path_train
        self.file_path_val = current_directory + configs.file_path_val
        self.file_path_test = current_directory + configs.file_path_test
        self.loadData()
        self.sess = tf.Session()

    def si_parse_fn(self, filename):
        """ Decode the image from the filename optional:convert to [0, 1]. """
        image_string_x = tf.read_file(filename[0])
        image_string_y = tf.read_file(filename[1])
        image_decoded_x = tf.image.decode_png(image_string_x, channels=3)
        image_decoded_y = tf.image.decode_png(image_string_y, channels=3)
        image_decoded = tf.concat([image_decoded_x, image_decoded_y], axis=2)
        return image_decoded  # returns concatenated x,y [H, W, 6]

    def si_opt_get_patches(self, image, num_crops_per_img, crop_size_h, crop_size_w):
        """ Get `num_crops_per_img` random crops from the image
         train set - full y size possible (to use 'fixed coord' y location - use x crop size)"""

        x_patches = []
        y_patches = []
        for i in range(num_crops_per_img):
            # crop x,y to max size y
            patch = tf.random_crop(image, [self.crop_size_h, self.crop_size_w, 6])

            # augment by left-right flips
            if self.do_flips:
                patch = tf.image.random_flip_left_right(patch)

            x, y_patch = tf.split(patch, 2, axis=2)

            # crop only x to patch size
            x_patch = tf.random_crop(x, [crop_size_h, crop_size_w, 3])

            y_patches.append(y_patch)
            x_patches.append(x_patch)

        y_patches = tf.stack(y_patches)
        x_patches = tf.stack(x_patches)

        # tuple of x _patches and y_images
        patches = tf.tuple([x_patches, y_patches])

        return patches

    def si_opt_crop_img(self, image, num_crops_per_img, crop_size_h, crop_size_w):

        """ Get `num_crops_per_img` crops from the center of image
        test set - full y size possible (to use 'fixed coord' y location - use x crop size)"""

        ImgSize = tf.shape(image)
        max_offset_h = (ImgSize[0] - self.crop_size_h) // 2
        max_offset_w = (ImgSize[1] - self.crop_size_w) // 2
        offset_h = (self.crop_size_h - crop_size_h) // 2
        offset_w = (self.crop_size_w - crop_size_w) // 2

        # center crop x,y to max size y
        x_patches = []
        y_patches = []
        for i in range(num_crops_per_img):
            # center crop x,y to max size y
            patch = tf.image.crop_to_bounding_box(image, max_offset_h, max_offset_w, self.crop_size_h, self.crop_size_w)

            x, y_patch = tf.split(patch, 2, axis=2)

            # center crop only x to patch size
            x_patch = tf.image.crop_to_bounding_box(x, offset_h, offset_w, crop_size_h, crop_size_w)

            y_patches.append(y_patch)
            x_patches.append(x_patch)

        y_patches = tf.stack(y_patches)
        x_patches = tf.stack(x_patches)

        # tuple of x _patches and y_images
        patches = tf.tuple([x_patches, y_patches])

        return patches

    def readfiles(self, fname):
        with open(fname) as f:
            content = f.readlines()
        content = [self.root_data + x.strip() for x in content]  # removes the `\n` at the end of each line
        return content

    def loadData(self):

        """using tf.data API the steps are:

            1. shuffle the filenames of the big images
            2. read the big images
            3. generate multiple patches from this image
            4. shuffle again all these patches with a big enough buffer size (see this answer on buffer size).
            Adjusting the buffer size is a tradeoff between good shuffling and size of the cached patches
            5. batch them
            6. repeat
            7. prefetch one batch"""

        get_patches_fn = lambda image: self.si_opt_get_patches(image, num_crops_per_img=self.num_crops_per_img,
                                                               crop_size_h=self.crop_size_h,
                                                               crop_size_w=self.crop_size_w)
        """ train dataset """
        content = self.readfiles(self.file_path_train)
        x_filenames_train = content[0:][::2]
        y_filenames_train = content[1:][::2]

        num_samples_train = len(x_filenames_train)

        # create tuple of paths lists (x1,y1) , (x2,y2), ...
        x_y_sets_train = list(zip(x_filenames_train, y_filenames_train))

        #  Create a Dataset serving batches of random patches in our images
        dataset_train = (tf.data.Dataset.from_tensor_slices(x_y_sets_train)
                         .shuffle(buffer_size=num_samples_train)
                         .map(self.si_parse_fn, num_parallel_calls=self.num_parallel_calls)
                         .map(get_patches_fn, num_parallel_calls=self.num_parallel_calls)
                         .apply(tf.contrib.data.unbatch())
                         .shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)
                         .batch(self.batch_size, drop_remainder=True)
                         .repeat(count=self.iterations)
                         .prefetch(1)
                         )
        iterator = dataset_train.make_one_shot_iterator()  # pass over dataset once
        self.x_train = iterator.get_next()  # get next batch. shape [2, batch, crop_size_h, crop_size_w, 3]

        """ val dataset """
        content = self.readfiles(self.file_path_val)
        x_filenames_val = content[0:][::2]
        y_filenames_val = content[1:][::2]

        # create tuple of paths lists (x1,y1) , (x2,y2), ...
        x_y_sets_val = list(zip(x_filenames_val, y_filenames_val))
        self.val_imgs_names = x_y_sets_val

        crop_img_fn = lambda image: self.si_opt_crop_img(image, num_crops_per_img=1,
                                                         crop_size_h=self.crop_size_h,
                                                         crop_size_w=self.crop_size_w)

        dataset_val = (tf.data.Dataset.from_tensor_slices(x_y_sets_val)
                       .map(self.si_parse_fn, num_parallel_calls=self.num_parallel_calls)
                       .map(crop_img_fn, num_parallel_calls=self.num_parallel_calls)
                       .apply(tf.contrib.data.unbatch())
                       .batch(self.batch_size, drop_remainder=True)
                       .repeat(count=self.iterations)
                       .prefetch(1)
                       )
        iterator = dataset_val.make_one_shot_iterator()
        self.x_val = iterator.get_next()  # shape [2, batch, crop_size_h, crop_size_w, 3]

        """ test dataset """
        content = self.readfiles(self.file_path_test)
        x_filenames_test = content[0:][::2]
        y_filenames_test = content[1:][::2]

        # create tuple of paths lists (x1,y1) , (x2,y2), ...
        x_y_sets_test = list(zip(x_filenames_test, y_filenames_test))
        self.test_imgs_names = x_y_sets_test

        dataset_test = (tf.data.Dataset.from_tensor_slices(x_y_sets_test)
            .map(self.si_parse_fn, num_parallel_calls=self.num_parallel_calls)
            .map(crop_img_fn, num_parallel_calls=self.num_parallel_calls)
            .apply(tf.contrib.data.unbatch())
            .batch(self.batch_size_test, drop_remainder=True)
            .repeat(count=self.iterations)
            .prefetch(1)
        )
        iterator = dataset_test.make_one_shot_iterator()
        self.x_test = iterator.get_next()  # shape [2, batch, crop_size_h, crop_size_w, 3]

    def get_data_size(self):
        return self.val_imgs_names, self.test_imgs_names

    def get_data_for_train(self):
        x = self.sess.run(self.x_train)
        return [np.transpose(x[0], (0,3,1,2)), np.transpose(x[1], (0,3,1,2))]

    def get_data_for_val(self):
        x = self.sess.run(self.x_val)
        return [np.transpose(x[0], (0,3,1,2)), np.transpose(x[1], (0,3,1,2))]

    def get_data_for_test(self):
        x = self.sess.run(self.x_test)
        return [np.transpose(x[0], (0, 3, 1, 2)), np.transpose(x[1], (0, 3, 1, 2))]
