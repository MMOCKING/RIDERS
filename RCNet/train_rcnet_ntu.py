from rcnet_main import train
import datetime
import os

class Config:
    def __init__(self):
        # Training and validation input filepaths
        self.root = '/media/lh/lh1/dataset/NTU'
        self.train_scenes = ['loop1_2022-06-03_0', 'loop1_2022-06-03_1', 'loop1_2022-06-03_2',
                            'loop1_2022-06-03_3', 'loop1_2022-06-03_4',
                            'loop2_2022-06-03_0', 'loop2_2022-06-03_2', 'loop2_2022-06-03_3',
                            'loop3_2022-06-03_1', 'loop3_2022-06-03_2',
                            ]

        self.image_file = 'thermal_undistort'
        self.radar_file = 'radar_png'
        self.gt_file = 'lidar_png_int'

        # Checkpoint and summary settings
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dirpath = \
            os.path.join('/media/lh/lh1/dataset/NTU/log/rcnet',current_time)
        self.n_step_per_checkpoint = 2000
        self.n_step_per_summary = 100
        self.restore_path = ''

        # Input settings
        self.batch_size = 24
        self.patch_size = [150, 50]
        self.total_points_sampled = 40
        self.sample_probability_lidar = 0.10
        self.input_channels_image = 3
        self.input_channels_depth = 3
        self.normalized_image_range = [0, 1]

        # Network settings
        self.encoder_type = ['rcnet', 'batch_norm']
        self.n_filters_encoder_image = [32, 64, 128, 128, 128]
        self.n_neurons_encoder_depth = [32, 64, 128, 128, 128]
        self.decoder_type = ['multiscale', 'batch_norm']
        self.n_filters_decoder = [256, 128, 64, 32, 16]

        # Weight settings
        self.weight_initializer = 'kaiming_uniform'
        self.activation_func = 'leaky_relu'

        # Training Settings
        self.learning_rates = [2e-4]
        self.learning_schedule = [200]

        # Augmentation Settings
        self.augmentation_probabilities = [1.00]
        self.augmentation_schedule = [-1]
        self.augmentation_random_brightness = [0.80, 1.20]
        self.augmentation_random_contrast = [0.80, 1.20]
        self.augmentation_random_saturation = [0.80, 1.20]
        self.augmentation_random_noise_type = ['none']
        self.augmentation_random_noise_spread = -1
        self.augmentation_random_flip_type = ['horizontal']

        # Loss settings
        self.w_weight_decay = 0.0
        self.w_positive_class = 2.5
        self.max_distance_correspondence = 0.5
        self.set_invalid_to_negative_class = False

        # Evaluation settings
        self.min_evaluate_depth = 0.0
        self.max_evaluate_depth = 100.0

        # Hardware
        self.n_thread = 8


# 创建配置实例
args = Config()

if __name__ == '__main__':

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    train(root=args.root,
          scenes=args.train_scenes,
          image_file=args.image_file,
          radar_file=args.radar_file,
          gt_file=args.gt_file,
          # Input settings
          batch_size=args.batch_size,
          patch_size=args.patch_size,
          total_points_sampled=args.total_points_sampled,
          sample_probability_of_lidar=args.sample_probability_lidar,
          normalized_image_range=args.normalized_image_range,
          # Network settings
          encoder_type=args.encoder_type,
          n_filters_encoder_image=args.n_filters_encoder_image,
          n_neurons_encoder_depth=args.n_neurons_encoder_depth,
          decoder_type=args.decoder_type,
          n_filters_decoder=args.n_filters_decoder,
          # Weight settings
          weight_initializer=args.weight_initializer,
          activation_func=args.activation_func,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_saturation=args.augmentation_random_saturation,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          # Loss settings
          w_weight_decay=args.w_weight_decay,
          w_positive_class=args.w_positive_class,
          max_distance_correspondence=args.max_distance_correspondence,
          set_invalid_to_negative_class=args.set_invalid_to_negative_class,
          # Checkpoint and summary settings
          checkpoint_dirpath=args.checkpoint_dirpath,
          n_step_per_summary=args.n_step_per_summary,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          restore_path=args.restore_path,
          # Hardware settings
          n_thread=args.n_thread)
