import numpy as np
import torch
# from models_old import GeneratorUNet, Discriminator

from data_loader_camus import DatasetCAMUS
from torchvision.utils import save_image
import metrics
from apex import amp
import math
# from torchsummary import summary
import datetime
import time
import sys
import random
import os
from utils import weights_init

RESULT_DIR = 'results'
VAL_DIR = 'val_images'
TEST_DIR = 'test_images'
MODELS_DIR = 'saved_models'

SEED = 17


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


class GAN:
    def __init__(self, config, use_wandb, device, dataset_path):

        # Configure data loader
        self.config = config
        self.result_name = config['NAME']

        self.use_wandb = use_wandb
        self.device = device
        self.epochs = config['EPOCHS']
        self.log_interval = config['LOG_INTERVAL']
        self.step = 0
        self.loaded_epoch = 0
        self.epoch = 0
        self.base_dir = './'
        self.model = config['MODEL']  # 0 for old model with nn.ConvTranspose2d, 1 for model with Pixel Shuffle

        self.patch = (1, config['PATCH_SIZE'], config['PATCH_SIZE'])
        # self.patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

        # Input shape
        self.channels = config['CHANNELS']
        self.img_rows = config['IMAGE_RES'][0]
        self.img_cols = config['IMAGE_RES'][1]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        assert self.img_rows == self.img_cols, 'The current code only works with same values for img_rows and img_cols'

        # Input images and their conditioning images

        self.conditional_d = config.get('CONDITIONAL_DISCRIMINATOR', False)
        self.recon_loss = config.get('RECON_LOSS', 'basic')
        self.loss_weight_d = config["LOSS_WEIGHT_DISC"]
        self.loss_weight_g = config["LOSS_WEIGHT_GEN"]

        # Calculate output shape of D (PatchGAN)
        patch_size = config['PATCH_SIZE']
        patch_per_dim = int(self.img_rows / patch_size)
        self.num_patches = (patch_per_dim, patch_per_dim, 1)

        # Number of filters in the first layer of G and D
        self.gf = config['FIRST_LAYERS_FILTERS']
        self.df = config['FIRST_LAYERS_FILTERS']
        self.skipconnections_generator = config['SKIP_CONNECTIONS_GENERATOR']
        self.output_activation = config['GEN_OUTPUT_ACT']
        self.decay_factor_G = config['LR_EXP_DECAY_FACTOR_G']
        self.decay_factor_D = config['LR_EXP_DECAY_FACTOR_D']

        if self.model == 1:
            from models import GeneratorUNet, Discriminator
            print('model 1 with Pixel Shuffle layers')
        elif self.model == 0:
            from models_old import GeneratorUNet, Discriminator
            print('model 0 with ConvTranspose2d layers')
        elif self.model == 2:
            from model_SRGAN import Generator, Discriminator
            print('model 2 SRGAN')
        elif self.model == 3:
            from model_SRGAN_2 import Generator, Discriminator, FeatureExtractor
            print('model 3 SRGAN ResNet VGG')

        # GeneratorUNet
        # self.generator = Generator(in_channels=self.channels, out_channels=self.channels).to(self.device)
        # self.discriminator = Discriminator(img_size=(self.img_rows, self.img_cols), in_channels=self.channels,
        #                                   patch_size=(patch_size, patch_size)).to(self.device)
        self.generator = Generator(scale_factor=1).to(self.device)
        #self.discriminator = Discriminator().to(self.device)
        #self.generator = Generator(in_channels=1, out_channels=1, n_residual_blocks=16).to(self.device)

        self.discriminator = Discriminator(input_shape=self.img_shape).to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=config['LEARNING_RATE_G'],
                                            betas=(config['ADAM_B1'], 0.999))  # 0.0002
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=config['LEARNING_RATE_D'],
                                            betas=(config['ADAM_B1'], 0.999))

        opt_level = 'O1'

        self.generator, self.optimizer_G = amp.initialize(self.generator,
                                                          self.optimizer_G,
                                                          opt_level=opt_level)

        self.discriminator, self.optimizer_D = amp.initialize(self.discriminator,
                                                              self.optimizer_D,
                                                              opt_level=opt_level)

        self.criterion_GAN = torch.nn.MSELoss().to(self.device)

        #self.feature_extractor.to(self.device)

        self.criterion_pixelwise = torch.nn.L1Loss(reduction='none').to(self.device)  # MAE
        # self.criterion_pixelwise = torch.nn.BCEWithLogitsLoss().to(self.device) # + weight + mean
        self.criterion_content = torch.nn.L1Loss().to(self.device)
        self.augmentation = dict()
        for key, value in config.items():
            if 'AUG_' in key:
                self.augmentation[key] = value

        self.train_data = DatasetCAMUS(dataset_path=dataset_path,
                                       random_state=config['RANDOM_SEED'],
                                       img_size=config['IMAGE_RES'],
                                       classes=config['LABELS'],
                                       train_ratio=config['TRAIN_RATIO'],
                                       valid_ratio=config['VALID_RATIO'],
                                       heart_states=config['HEART_STATES'],
                                       views=config['HEART_VIEWS'],
                                       image_qualities=config['IMAGE_QUALITIES'],
                                       patient_qualities=config['PATIENT_QUALITIES'],
                                       # augment=self.augmentation,
                                       subset='train')
        self.valid_data = DatasetCAMUS(dataset_path=dataset_path,
                                       random_state=config['RANDOM_SEED'],
                                       img_size=config['IMAGE_RES'],
                                       classes=config['LABELS'],
                                       train_ratio=config['TRAIN_RATIO'],
                                       valid_ratio=config['VALID_RATIO'],
                                       heart_states=config['HEART_STATES'],
                                       views=config['HEART_VIEWS'],
                                       image_qualities=config['IMAGE_QUALITIES'],
                                       patient_qualities=config['PATIENT_QUALITIES'],
                                       # augment=self.augmentation,
                                       subset='valid')

        self.test_data = DatasetCAMUS(dataset_path=dataset_path,
                                      random_state=config['RANDOM_SEED'],
                                      img_size=config['IMAGE_RES'],
                                      classes=config['LABELS'],
                                      train_ratio=config['TRAIN_RATIO'],
                                      valid_ratio=config['VALID_RATIO'],
                                      heart_states=config['HEART_STATES'],
                                      views=config['HEART_VIEWS'],
                                      image_qualities=config['IMAGE_QUALITIES'],
                                      patient_qualities=config['PATIENT_QUALITIES'],
                                      # augment=self.augmentation,
                                      subset='test')

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=config['BATCH_SIZE'],  # 32 max
                                                        shuffle=True,
                                                        num_workers=config['NUM_WORKERS'])
        self.valid_loader = torch.utils.data.DataLoader(self.valid_data,
                                                        batch_size=config['BATCH_SIZE'],
                                                        shuffle=False,
                                                        num_workers=config['NUM_WORKERS'])
        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=config['BATCH_SIZE'],
                                                       shuffle=False,
                                                       num_workers=config['NUM_WORKERS'])

        # Training hyper-parameters
        self.batch_size = config['BATCH_SIZE']
        # self.max_iter = config['MAX_ITER']
        self.val_interval = config['VAL_INTERVAL']
        self.log_interval = config['LOG_INTERVAL']
        self.save_model_interval = config['SAVE_MODEL_INTERVAL']
        self.lr_G = config['LEARNING_RATE_G']
        self.lr_D = config['LEARNING_RATE_D']

        self.average_loss_G_train = metrics.AverageMeter()
        self.average_loss_G_valid = metrics.AverageMeter()

    # do not use
    def valid(self):
        # validation
        self.generator.eval()
        self.discriminator.eval()
        # for i, batch in enumerate(self.valid_loader):
        #    image, mask, full_mask, weight_map, segment_mask, quality, heart_state, view = batch
        image, mask, full_mask, weight_map, segment_mask, quality, heart_state, view = next(iter(self.valid_loader))
        mask = mask.to(self.device)
        image = image.to(self.device)
        full_mask = full_mask.to(self.device)
        weight_map = weight_map.to(self.device)

        # Adversarial ground truths for discriminator losses
        #patch_real = torch.tensor(np.ones((mask.size(0), *self.patch)), dtype=torch.float32, device=self.device)
        #patch_fake = torch.tensor(np.zeros((mask.size(0), *self.patch)), dtype=torch.float32,
        #                          device=self.device)

        fake_echo = self.generator(full_mask)  # * segment_mask  # mask

        # Real loss
        pred_real = self.discriminator(image, mask)
        # loss_real = self.criterion_GAN(pred_real, patch_real)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake loss
        pred_fake = self.discriminator(fake_echo.detach(), mask)
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_GAN = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        # loss_GAN = 0

        # Pixel-wise loss
        loss_pixel = torch.mean(self.criterion_pixelwise(fake_echo, image) * weight_map)  # * segment_mask

        # Total loss
        loss_G = self.loss_weight_d * loss_GAN + self.loss_weight_g * loss_pixel  # 1 100

        psnr = metrics.psnr(mask, fake_echo)  # * segment_mask
        ssim = metrics.ssim(mask, fake_echo, window_size=11, size_average=True)  # * segment_mask
        return loss_D.item(), loss_fake.item(), loss_real.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item(), psnr, ssim

    def train(self):

        prev_time = time.time()
        batch_size = self.batch_size
        # max_iter = self.max_iter
        val_interval = self.val_interval
        log_interval = self.log_interval
        save_model_interval = self.save_model_interval

        for epoch in range(self.loaded_epoch, self.epochs):
            self.epoch = epoch
            for i, batch in enumerate(self.train_loader):

                image, mask, full_mask, weight_map, segment_mask, quality, heart_state, view = batch

                mask = mask.to(self.device)
                image = image.to(self.device)
                full_mask = full_mask.to(self.device)
                weight_map = weight_map.to(self.device)
                segment_mask = segment_mask.to(self.device)

                # Adversarial ground truths for discriminator losses
                # patch_real = torch.tensor(np.ones((mask.size(0), *self.patch)), dtype=torch.float32, device=self.device)
                # patch_fake = torch.tensor(np.zeros((mask.size(0), *self.patch)), dtype=torch.float32,
                #                          device=self.device)

                #  Train Discriminator

                self.generator.eval()
                self.discriminator.train()

                self.optimizer_D.zero_grad()

                fake_echo = self.generator(full_mask)  # * segment_mask  # mask

                # Real loss
                pred_real = self.discriminator(image, mask)
                # loss_real = self.criterion_GAN(pred_real, patch_real)
                loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

                # Fake loss

                pred_fake = self.discriminator(fake_echo.detach(), mask)
                loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_real))

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer_D.step()

                #  Train Generator

                self.generator.train()
                self.discriminator.eval()

                self.optimizer_G.zero_grad()

                # GAN loss
                fake_echo = self.generator(full_mask)
                pred_fake = self.discriminator(fake_echo, mask)

                loss_GAN = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

                # Content Loss
                gen_features = self.feature_extractor(fake_echo)
                real_features = self.feature_extractor(image)
                loss_content = self.criterion_content(gen_features, real_features.detach())

                # Pixel-wise loss
                #loss_pixel = torch.mean(self.criterion_pixelwise(fake_echo, image) * weight_map)  # * segment_mask

                # Total loss
                loss_G = self.loss_weight_d * loss_GAN + self.loss_weight_g * loss_content  # 1 100

                with amp.scale_loss(loss_G, self.optimizer_G) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer_G.step()

                #  Log Progress

                # Determine approximate time left
                batches_done = self.epoch * len(self.train_loader) + i
                batches_left = self.epochs * len(self.train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # metrics
                psnr = metrics.psnr(mask, fake_echo)  # * segment_mask
                ssim = metrics.ssim(mask, fake_echo, window_size=11, size_average=True)  # * segment_mask

                # self.average_loss_G_train.update(loss_G.item())
                # print(self.average_loss_G_train.val)

                # print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f patch_fake: %f real: %f] [G loss: %f, pixel: %f, adv: %f] PSNR: %f SSIM: %f ETA: %s"
                    % (
                        self.epoch,
                        self.epochs,
                        i,
                        len(self.train_loader),
                        loss_D.item(),
                        loss_fake.item(),
                        loss_real.item(),
                        loss_G.item(),
                        loss_content.item(),
                        loss_GAN.item(),
                        psnr,
                        ssim,
                        time_left,
                    )
                )
                # sys.stdout.write(
                #    "\r[Epoch %d/%d] [Batch %d/%d] TRAIN [D loss: %f G loss: %f PSNR: %f SSIM: %f] VAL [D loss: %f G loss: %f PSNR: %f SSIM: %f]" % (
                #        self.epoch,
                #        self.epochs,
                #        i,
                #        len(self.train_loader),
                #        loss_D.item(),
                #        loss_G.item(),
                #        psnr,
                #        ssim,
                #        loss_D_val,
                #        loss_G_val,
                #        psnr_val,
                #        ssim_val

                #    )
                #    )

                # save images
                if batches_done % self.log_interval == 0:
                    self.generator.eval()
                    self.discriminator.eval()
                    self.sample_images(batches_done)

                # log wandb
                self.step += 1
                if self.use_wandb:
                    import wandb
                    wandb.log({'loss_D': loss_D, 'loss_real_D': loss_real, 'loss_fake_D': loss_fake,
                               'loss_G': loss_G, 'loss_pixel': loss_content, 'loss_GAN': loss_GAN,
                               'PSNR': psnr, 'SSIM': ssim,
                               # 'loss_D_val': loss_D_val, 'loss_real_D_val': loss_real_val,
                               # 'loss_fake_D_val': loss_fake_val,
                               # 'loss_G_val': loss_G_val, 'loss_pixel_val': loss_pixel_val, 'loss_GAN_val': loss_GAN_val,
                               # 'PSNR_val': psnr_val, 'SSIM_val': ssim_val
                               },

                              step=self.step)

            # save models
            if (epoch + 1) % save_model_interval == 0:
                self.save(f'{self.base_dir}/generator_last_checkpoint.bin', model='generator')
                self.save(f'{self.base_dir}/discriminator_last_checkpoint.bin', model='discriminator')
        loss_D_val, loss_fake_val, loss_real_val, loss_G_val, loss_pixel_val, loss_GAN_val, psnr_val, ssim_val = self.valid()
        print(loss_D_val, loss_fake_val, loss_real_val, loss_G_val, loss_pixel_val, loss_GAN_val, psnr_val, ssim_val)

    def sample_images(self, batches_done):
        """Saves a generated sample from the validation set"""
        image, mask, full_mask, weight_map, segment_mask, quality, heart_state, view = next(iter(self.valid_loader))
        image = image.to(self.device)
        mask = mask.to(self.device)
        full_mask = full_mask.to(self.device)
        quality = quality.to(self.device)
        segment_mask = segment_mask.to(self.device)
        fake_echo = self.generator(full_mask)  # * segment_mask # , quality)
        img_sample = torch.cat((image.data, fake_echo.data, mask.data), -2)
        save_image(img_sample, "images/%s.png" % batches_done, nrow=4, normalize=True)

        batch = 4
        image = image.cpu().detach().numpy()
        fake_echo = fake_echo.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        quality = quality.cpu().detach().numpy()

        img_sample = np.concatenate([image,
                                     fake_echo,
                                     mask], axis=1)
        q = ['low', 'med', 'high']
        import matplotlib.pyplot as plt
        rows, cols = 3, batch
        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for row in range(rows):
            for col in range(cols):
                class_label = np.argmax(quality[col], axis=1)[0]

                axs[row, col].imshow(img_sample[col, row, :, :], cmap='gray')
                axs[row, col].set_title(titles[row] + ' ' + q[class_label], fontdict={'fontsize': 6})
                axs[row, col].axis('off')
                cnt += 1

        # fig.savefig('%s/%s/%s/%s_%d.png' % (RESULT_DIR, self.result_name, VAL_DIR, prefix, step_num))
        fig.savefig("images/_%s.png" % batches_done)
        plt.close(fig)

        if self.use_wandb:
            import wandb
            wandb.log({'val_image': fig}, step=self.step)

    def save(self, path, model='generator'):
        if model == 'generator':
            self.generator.eval()
            torch.save({
                'model_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': self.optimizer_G.state_dict(),

                # 'scheduler_state_dict': self.scheduler.state_dict(),
                # 'best_summary_loss': self.best_summary_loss,
                'epoch': self.epoch,
            }, path)
            # print('\ngenerator saved, epoch ', self.epoch)
        elif model == 'discriminator':

            self.discriminator.eval()
            torch.save({
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_D.state_dict(),

                # 'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler_state_dict': self.scheduler.state_dict(),
                # 'best_summary_loss': self.best_summary_loss,
                'epoch': self.epoch,
            }, path)
            # print('discriminator saved, epoch ', self.epoch)

    def load(self, path, model='generator'):
        if model == 'generator':
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # self.best_summary_loss = checkpoint['best_summary_loss']
            self.loaded_epoch = checkpoint['epoch'] + 1
            print('generator loaded, epoch ', self.loaded_epoch)
        elif model == 'discriminator':
            checkpoint = torch.load(path)
            self.discriminator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loaded_epoch = checkpoint['epoch'] + 1
            print('discriminator loaded, epoch ', self.loaded_epoch)

            # self.best_summary_loss = checkpoint['best_summary_loss']
