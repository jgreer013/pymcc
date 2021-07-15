import matplotlib.pyplot as plt
import neural_net
import numpy as np
import pandas as pd
import adabound
import os
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.io as tvio
import torchvision.models as models
import kornia

from hyperdash import Experiment
from skimage import io, transform
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image

DIR = "./samples/2021_03_23_20_28_39/"
CSV_FILENAME = "data_sorted.csv"
#PATH = "model.pt"
#PATH = "model_shifted_adam_0_01_e20.pt"
PATH = "WGANGP_withPrevAction_NEW_was_mse_30_sgd_16_0_01_e50_ncritic_1.pt"
SEED = 13
#VAL_SIZE = 5000
#VAL_SIZE = 16500
#VAL_SIZE = 53445 # 100 batches
VAL_SIZE = 54945
#VAL_SIZE = 55013 # 2 batches
#VAL_SIZE = 55043
#VAL_SIZE = 54981
NUM_EPOCHS = 200
BATCH_SIZE = 2
USE_HALF = False
USE_GPU = True
#REAL_LABEL = 1
REAL_LABEL = -1
#FAKE_LABEL = 0
FAKE_LABEL = 1
LEARNING_RATE = 0.0006
BETA_ONE = 0.5
BETA_TWO = 0.999
N_CRITIC = 5
LAMBDA_GP = 10
LAMBDA_L2 = 30

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def show_image(image):
    """Show image"""
    plt.imshow(image, aspect="auto")
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_sample_images(halo_dataset):
    fig = plt.figure()

    for i in range(len(halo_dataset)):
        sample = halo_dataset[i]

        print(i, sample['image'].shape, sample['controller_state'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_image(sample['image'].permute(1, 2, 0))

        if i == 3:
            plt.show()
            break

class HaloReachDataset(Dataset):
    """Halo Reach dataset."""

    def __init__(self, csv_file_name, root_dir, transform=None, use_gpu=True, use_half=True):
        """
        Args:
            csv_file (string): Path to the csv file with filenames and controller states.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.halo_frame = pd.read_csv(os.path.join(root_dir, csv_file_name), header=None)

        
        cols = self.halo_frame.columns
        # round triggers to nearest integer
        self.halo_frame[cols[5:7]] = self.halo_frame[cols[5:7]].round(0)

        # normalize triggers and buttons to -1 to 1
        self.halo_frame[cols[5:21]] = ((self.halo_frame[cols[5:21]] * 2.0) - 1.0).clip(-1, 1)

        # Assign current action
        self.current_controller_state = self.halo_frame[cols[1:21]]

        # shift actions by 1
        self.halo_frame[cols[1:21]] = self.halo_frame[cols[1:21]].shift(-1)
        
        # normalize sticks to 0 - 1
        #self.halo_frame[cols[1:5]] = ((self.halo_frame[cols[1:5]] + 1) / 2.0).clip(0, 1)

        

        # CLASSIFICATION
        # Convert stick values to values of [0, 1, 2, 3, 4] based on bins of [-1.1, -0.75, -0.25, 0.25, 0.75, 1.1]
        #bins = [-1.1, -0.75, -0.25, 0.25, 0.75, 1.1]
        #for i in range(1,5):
        #    self.halo_frame[[i]] = pd.cut(self.halo_frame[[i]].values.flatten(), bins, labels=[0,1,2,3,4]).codes

        

        

        # Remove last row as it has no subsequent action data
        self.halo_frame = self.halo_frame[:-1]
        self.current_controller_state = self.current_controller_state[:-1]
        
        self.root_dir = root_dir
        self.transform = transform

        if use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        self.use_half = use_half

    def __len__(self):
        return len(self.halo_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                os.path.basename(self.halo_frame.iloc[idx, 0]))
        image = Image.open(img_name)
        next_controller_state = self.halo_frame.iloc[idx, 1:21]
        next_controller_state = np.array([next_controller_state])
        next_controller_state = next_controller_state.astype('float').flatten()

        current_controller_state = self.current_controller_state.iloc[idx, :]
        current_controller_state = np.array([current_controller_state])
        current_controller_state = current_controller_state.astype("float").flatten()
        image = image.resize((960, 540))

        sample = {'image': image, 'next_controller_state': next_controller_state, 'current_controller_state': current_controller_state, 'file_name': os.path.basename(self.halo_frame.iloc[idx, 0])}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
def compute_gradient_penalty(D, real_samples, fake_samples, image_samples, prev_actions):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(image_samples, interpolates, prev_actions)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return LAMBDA_GP * gradient_penalty

def train(train_set, val_set, optimizer_g, optimizer_d, criterion, generator, discriminator, experiment, save_on_val_perf=True):
    print_step = 5
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    if USE_GPU:
        one = one.cuda()
        mone = mone.cuda()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
    generator.train()
    discriminator.train()
    best_average_validation_loss = 100
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, factor=0.5, patience=1, verbose=True)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        print("Epoch: ", epoch + 1)
        experiment.metric("Epoch", epoch + 1)
        running_loss_g = 0.0
        running_loss_g_mse = 0.0
        running_loss_d = 0.0
        running_avg_wass_d = 0.0
        batch_count = 0
        for i, data in enumerate(trainloader, 0):
            batch_count += 1
            images, prev_actions, real_actions = data["image"].float(), data["current_controller_state"].float(), data["next_controller_state"].float()
            batch_size = images.size(0)
            if USE_GPU:
                images = images.cuda()
                real_actions = real_actions.cuda()
                prev_actions = prev_actions.cuda()
            images_v = Variable(images)
            real_actions_v = Variable(real_actions)
            prev_actions_v = Variable(prev_actions)

            ### Update Discriminator ###
            for p in discriminator.parameters():
                p.requires_grad = True

            optimizer_d.zero_grad()

            # Train with real batch
            discriminator_output_real = discriminator(images_v, real_actions_v, prev_actions_v)

            # Train with fake batch
            #sticks, button_probs = generator(images, prev_actions)
            #fake_actions = torch.cat((sticks, button_probs), dim=1)
            fake_actions = generator(images, prev_actions)
            discriminator_output_fake = discriminator(images_v, fake_actions, prev_actions_v)
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_actions_v.data, fake_actions.data, images_v.data, prev_actions_v.data)

            d_loss = -torch.mean(discriminator_output_real) + torch.mean(discriminator_output_fake) + gradient_penalty
            d_loss.backward()
            optimizer_d.step()

            running_loss_d += d_loss.item()

            optimizer_g.zero_grad()

            if i % print_step == (print_step-1):
                experiment.metric("RM Train Loss Dis", running_loss_d / print_step)
                #experiment.metric("RM Was Train Dist", running_avg_wass_d / print_step)
                running_loss_d = 0.0
                running_avg_wass_d = 0.0

            if i % N_CRITIC == N_CRITIC - 1:
                ### Update Generator ###
                # Fake labels are real for generator cost
                # Needed?
                #sticks, button_probs = generator(images)
                #fake_actions = torch.cat((sticks, button_probs), dim=1)
                for p in discriminator.parameters():
                    p.requires_grad = False

                fake_actions = generator(images_v.data, prev_actions_v.data)

                # Discriminator has been updated, pass through again
                discriminator_output_gen = discriminator(images_v.data, fake_actions, prev_actions_v.data)
                
                g_loss = -torch.mean(discriminator_output_gen)
                g_loss.backward()
                optimizer_g.step()
                print(fake_actions)
                print(real_actions_v)
                l2_loss = criterion(fake_actions.data, real_actions_v.data)
                print(l2_loss)
                #g_loss = gen_loss #+ (LAMBDA_L2 * l2_loss)
                

                # print statistics
                running_loss_g += g_loss.item()
                #running_loss_g_mse = LAMBDA_L2 * l2_loss.item()
                #running_loss_d += err_d.item()
                if i % print_step == (print_step-1):
                    experiment.metric("RM Train Loss Gen", running_loss_g / (print_step / N_CRITIC))
                    #experiment.metric("RM Train Loss Gen MSE", running_loss_g_mse / (print_step / N_CRITIC))
                    print('[%d, %5d, samples: %d] generator loss: %.8f' %
                        (epoch + 1, i + 1, (i+1) * BATCH_SIZE, running_loss_g / (print_step / N_CRITIC)))
                    running_loss_g = 0.0
                    running_loss_g_mse = 0.0

        """
        # Check validation once per epoch
        if save_on_val_perf:
            avg_validation_loss, max_validation_loss = validate(val_set, generator, discriminator)
            scheduler.step(avg_validation_loss)
            experiment.metric("Average Validation Loss", avg_validation_loss)
            experiment.metric("Max Validation Loss", max_validation_loss)
            generator.train()
            discriminator.train()
            if (avg_validation_loss < best_average_validation_loss):
                best_average_validation_loss = avg_validation_loss
                print("Saving...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer_d.state_dict(),
                    }, PATH)
            else:
                print("Loss on validation set did not improve, skipping save to avoid overfitting")
        else:
            print("Saving...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_d.state_dict(),
                }, PATH)
        """

    print('Finished Training')

def validate(val_set, generator, discriminator):
    valloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=5)
    generator.eval()
    discriminator.eval()

    max_loss = -100
    max_loss_index = -1
    max_loss_label = None
    max_loss_prediction = None
    max_loss_file_name = ""

    for epoch in range(1):
        print("Epoch: ", epoch + 1)
        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(valloader, 0):
            images, prev_actions, real_actions = data["image"].float(), data["current_controller_state"].float(), data["next_controller_state"].float()
            batch_size = images.size(0)
            if USE_GPU:
                images = images.cuda()
                real_actions = real_actions.cuda()
                prev_actions = prev_actions.cuda()

            # Fake labels are real for generator cost
            discriminator_output_real = discriminator(images, real_actions, prev_actions)

            # Train with fake batch
            #sticks, button_probs = generator(images, prev_actions)
            #fake_actions = torch.cat((sticks, button_probs), dim=1)
            fake_actions = generator(images, prev_actions)
            discriminator_output_fake = discriminator(images, fake_actions, prev_actions)

            d_loss = -torch.mean(discriminator_output_real) + torch.mean(discriminator_output_fake)
            loss = d_loss.item()

            if (loss > max_loss):
                print("Found new max loss %.8f for file %s" % (loss, data["file_name"]))
                max_loss_index = i
                max_loss_label = real_actions
                max_loss_prediction = fake_actions
                max_loss = loss
                max_loss_file_name = data["file_name"]

            # print statistics
            running_loss += loss
            total_loss += loss
            if i % 1000 == 999:
                print('[%d, %5d, samples: %d] loss: %.8f' %
                    (epoch + 1, i + 1, (i + 1) * 1, running_loss / (i+1)))
                running_loss = 0

    avg_loss = total_loss / len(val_set)
    print("Average loss: %.8f" % avg_loss)
    print("Sample with largest loss:")
    print("Sample: %d" % max_loss_index)
    print("Loss: %.8f" % max_loss)
    print("Expected: ", max_loss_label)
    print("Actual: ", max_loss_prediction)
    print("Filename: ", max_loss_file_name)

    return avg_loss, max_loss

if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Normalization for ResNet
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    """
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    halo_dataset = HaloReachDataset(csv_file_name=CSV_FILENAME, root_dir=DIR, use_gpu=USE_GPU, use_half=USE_HALF, transform=composed_transforms)
    generator_net = neural_net.GeneratorWithAction()
    discriminator_net = neural_net.ResnetImageActionDiscriminatorWGANGPWithAction()

    gpu = torch.device("cuda:0")
    if USE_GPU:
        generator_net.to(gpu)
        discriminator_net.to(gpu)
    if USE_HALF:
        generator_net.half()
        discriminator_net.half()

    #criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    #optimizer_g = optim.SGD(generator_net.parameters(), lr=LEARNING_RATE)
    #optimizer_d = optim.SGD(discriminator_net.parameters(), lr=LEARNING_RATE)
    optimizer_g = optim.Adam(generator_net.parameters(), lr=LEARNING_RATE, betas=(BETA_ONE, BETA_TWO))
    optimizer_d = optim.Adam(discriminator_net.parameters(), lr=LEARNING_RATE, betas=(BETA_ONE, BETA_TWO))
    #optimizer_g = optim.RMSprop(generator_net.parameters(), lr=LEARNING_RATE)
    #optimizer_d = optim.RMSprop(discriminator_net.parameters(), lr=LEARNING_RATE)

    train_set, val_set = torch.utils.data.random_split(halo_dataset, [len(halo_dataset) - VAL_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(SEED))

    exp = Experiment("Halo AI - ResNet18 - WGAN-GP - With Prev Action")
    exp.param("Optimizer", "SGD")
    exp.param("learning rate", LEARNING_RATE)
    exp.param("Scheduler", "ReduceLROnPlateau")
    exp.param("loss", "Wasserstein + MSE 30")
    exp.param("Validation Set Size", len(val_set))
    exp.param("Training Set Size", len(train_set))
    exp.param("Batch Size", BATCH_SIZE)
    train(train_set, val_set, optimizer_g, optimizer_d, criterion, generator_net, discriminator_net, exp, save_on_val_perf=True)
    exp.end()



    

