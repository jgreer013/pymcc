import matplotlib.pyplot as plt
import neural_net
import numpy as np
import pandas as pd
import adabound
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.io as tvio
import torchvision.models as models
import kornia

from hyperdash import Experiment
from skimage import io, transform
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image

DIR = "./samples/2021_03_23_20_28_39/"
CSV_FILENAME = "data_sorted.csv"
#PATH = "model.pt"
#PATH = "model_shifted_adam_0_01_e20.pt"
PATH = "GeneratorWithAction_mse_adam_8_0_01_e50.pt"
SEED = 13
#VAL_SIZE = 5000
#VAL_SIZE = 16500
#VAL_SIZE = 53445 # 100 batches
#VAL_SIZE = 55013 # 2 batches
#VAL_SIZE = 55043 # 1 sample
#VAL_SIZE = 54981
VAL_SIZE = 54945
NUM_EPOCHS = 50
BATCH_SIZE = 8
USE_HALF = False
USE_GPU = True
LEARNING_RATE = 0.01

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
        #self.halo_frame[cols[5:7]] = self.halo_frame[cols[5:7]].round(0)

        # normalize triggers and buttons to -1 to 1
        self.halo_frame[cols[5:21]] = ((self.halo_frame[cols[5:21]] * 2.0) - 1.0).clip(-1, 1)

        #self.halo_frame[cols] = self.halo_frame[cols] * 100

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

def train(train_set, val_set, optimizer, criterion, net, experiment, save_on_val_perf=True, extra_criterion=None, stick_log_var=None, button_log_var=None):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    net.train()
    print_step = 1
    best_average_validation_loss = 100
    previous_epoch_loss = -1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        print("Epoch: ", epoch + 1)
        experiment.metric("Epoch", epoch + 1)
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_count += 1
            inputs, current_controller_states, labels = data["image"].float(), data["current_controller_state"].float(), data["next_controller_state"].float()
            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
                current_controller_states = current_controller_states.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            #stick_labels, button_labels = torch.split(labels, [4, 16], dim=1)
            """
            stick_l_lr_labels, stick_l_ud_labels, stick_r_lr_labels, stick_r_ud_labels, button_labels = torch.split(labels, [1, 1, 1, 1, 16], dim=1)
            stick_l_lr_labels = stick_l_lr_labels.squeeze(1).long()
            stick_l_ud_labels = stick_l_ud_labels.squeeze(1).long()
            stick_r_lr_labels = stick_r_lr_labels.squeeze(1).long()
            stick_r_ud_labels = stick_r_ud_labels.squeeze(1).long()
            """

            # forward + backward + optimize
            #outputs = net(inputs)
            #sticks, buttons, button_probs = net(inputs, current_controller_states)
            #sticks, button_probs = net(inputs)
            #stick_l_lr, stick_l_ud, stick_r_lr, stick_r_ud, buttons, stick_l_lr_probs, stick_l_ud_probs, stick_r_lr_probs, stick_r_ud_probs, buttons_probs = net(inputs)
            #outputs = torch.tanh(net(inputs))
            output_action = net(inputs, current_controller_states)
            #output_action = net(inputs)
            """
            l_lr_loss = criterion(stick_l_lr, stick_l_lr_labels, alpha=0.25, gamma=2.0, reduction='mean')
            l_ud_loss = criterion(stick_l_ud, stick_l_ud_labels, alpha=0.25, gamma=2.0, reduction='mean')
            r_lr_loss = criterion(stick_r_lr, stick_r_lr_labels, alpha=0.25, gamma=2.0, reduction='mean')
            r_ud_loss = criterion(stick_r_ud, stick_r_ud_labels, alpha=0.25, gamma=2.0, reduction='mean')
            average_stick_loss = 0.25 * l_lr_loss + 0.25 * l_ud_loss + 0.25 * r_lr_loss + 0.25 * r_ud_loss
            loss = average_stick_loss
            """
            loss = criterion(output_action, labels)
            #stick_loss = criterion(sticks, stick_labels)
            #button_loss = extra_criterion(buttons, button_labels, alpha=0.25, gamma=2.0, reduction='mean')
            #button_loss = extra_criterion(button_probs, button_labels)
            #raw_summed_loss = stick_loss + button_loss
            #loss = uncertainty_criterion(stick_loss, button_loss, stick_log_var, button_log_var)
            #loss = (4 * stick_loss) + button_loss

            loss.backward()
            optimizer.step()
            #scheduler.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % print_step == (print_step - 1):
                print('[%d, %5d, samples: %d] loss: %.8f' %
                    (epoch + 1, i + 1, (i+1) * BATCH_SIZE, running_loss / print_step))
                #print("Raw summed loss: %.8f" % raw_summed_loss)
                experiment.metric("Running Average Training Loss", running_loss / print_step)
                running_loss = 0.0

        epoch_loss = epoch_loss / batch_count
        experiment.metric("Average Epoch Training Loss", epoch_loss)
        if previous_epoch_loss != -1:
            print("Change from last epoch: ", epoch_loss - previous_epoch_loss)
        previous_epoch_loss = epoch_loss

        """
        # Check validation once per epoch
        if save_on_val_perf:
            avg_validation_loss, max_validation_loss = validate(val_set, criterion, net, extra_criterion, stick_log_var, button_log_var)
            scheduler.step(avg_validation_loss)
            experiment.metric("Average Validation Loss", avg_validation_loss)
            experiment.metric("Max Validation Loss", max_validation_loss)
            net.train()
            if (avg_validation_loss < best_average_validation_loss):
                best_average_validation_loss = avg_validation_loss
                print("Saving...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, PATH)
            else:
                print("Loss on validation set did not improve, skipping save to avoid overfitting")
        else:
            print("Saving...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, PATH)
        """

    print('Finished Training')

def validate(val_set, criterion, net, extra_criterion=None, stick_log_var=None, button_log_var=None):
    valloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
    net.eval()

    max_loss = -1
    max_loss_index = -1
    max_loss_label = None
    max_loss_prediction = None
    max_loss_file_name = ""

    for epoch in range(1):
        print("Epoch: ", epoch + 1)
        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, current_controller_states, labels = data["image"].float(), data["current_controller_state"].float(), data["next_controller_state"].float()
            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
                current_controller_states = current_controller_states.cuda()

            #stick_labels, button_labels = torch.split(labels, [4, 16], dim=1)

            """
            stick_l_lr_labels, stick_l_ud_labels, stick_r_lr_labels, stick_r_ud_labels, button_labels = torch.split(labels, [1, 1, 1, 1, 16], dim=1)
            stick_l_lr_labels = stick_l_lr_labels.long().squeeze(1)
            stick_l_ud_labels = stick_l_ud_labels.long().squeeze(1)
            stick_r_lr_labels = stick_r_lr_labels.long().squeeze(1)
            stick_r_ud_labels = stick_r_ud_labels.long().squeeze(1)
            """
            # forward + backward + optimize
            #outputs = net(inputs)
            #stick_l_lr, stick_l_ud, stick_r_lr, stick_r_ud, buttons, stick_l_lr_probs, stick_l_ud_probs, stick_r_lr_probs, stick_r_ud_probs, buttons_probs = net(inputs)
            #sticks, buttons, button_probs = net(inputs, current_controller_states)
            #sticks, button_probs = net(inputs)
            output_action = net(inputs, current_controller_states)
            #output_action = net(inputs)
            #outputs = torch.tanh(net(inputs))
            """
            l_lr_loss = criterion(stick_l_lr, stick_l_lr_labels, alpha=0.25, gamma=2.0, reduction='mean')
            l_ud_loss = criterion(stick_l_ud, stick_l_ud_labels, alpha=0.25, gamma=2.0, reduction='mean')
            r_lr_loss = criterion(stick_r_lr, stick_r_lr_labels, alpha=0.25, gamma=2.0, reduction='mean')
            r_ud_loss = criterion(stick_r_ud, stick_r_ud_labels, alpha=0.25, gamma=2.0, reduction='mean')
            average_stick_loss = 0.25 * l_lr_loss + 0.25 * l_ud_loss + 0.25 * r_lr_loss + 0.25 * r_ud_loss
            loss = average_stick_loss
            """
            #stick_loss = criterion(sticks, stick_labels)
            #button_loss = extra_criterion(buttons, button_labels, alpha=0.25, gamma=2.0, reduction='mean')
            #button_loss = extra_criterion(button_probs, button_labels)
            #loss = uncertainty_criterion(stick_loss, button_loss, stick_log_var, button_log_var)
            #loss = (4 * stick_loss) + button_loss
            loss = criterion(output_action, labels)

            if (loss.item() > max_loss):
                print("Found new max loss %.8f for file %s" % (loss.item(), data["file_name"]))
                max_loss_index = i
                max_loss_label = labels
                #max_loss_label = (stick_l_lr_labels, stick_l_ud_labels, stick_r_lr_labels, stick_r_ud_labels, button_labels)
                #max_loss_label = (stick_labels, button_labels)
                max_loss_prediction = output_action
                #max_loss_prediction = (torch.argmax(stick_l_lr_probs, dim=1), torch.argmax(stick_l_ud_probs, dim=1), torch.argmax(stick_r_lr_probs, dim=1), torch.argmax(stick_r_ud_probs, dim=1), buttons_probs)
                #max_loss_prediction = (sticks, button_probs)
                max_loss = loss.item()
                max_loss_file_name = data["file_name"]

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d, samples: %d] loss: %.8f' %
                    (epoch + 1, i + 1, (i + 1) * 1, running_loss / 1000))
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

def uncertainty_criterion(stick_loss, button_loss, stick_log_var, button_log_var):
    loss = 0
    stick_precision = torch.exp(-stick_log_var)
    button_precision = torch.exp(-button_log_var)
    stick_uncertainty_loss = (0.5 * (stick_precision * stick_loss)) + stick_log_var
    button_uncertainty_loss = (button_precision * button_loss) + button_log_var
    loss = stick_uncertainty_loss + button_uncertainty_loss
    return loss

if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    file_data = pd.read_csv(os.path.join(DIR, CSV_FILENAME))
    file_names = file_data.iloc[:, 0]
    controller_states = file_data.iloc[:, 1:21]
    timestamps = file_data.iloc[:, 21]
    N = len(file_names)
    i = 11700

    img_name = os.path.basename(file_names.iloc[i])
    
    # Normalization for ResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        #normalize,
    ])
    halo_dataset = HaloReachDataset(csv_file_name=CSV_FILENAME, root_dir=DIR, use_gpu=USE_GPU, use_half=USE_HALF, transform=composed_transforms)

    ### Uncomment to show sample images in plot ###
    #fig = plt.figure(dpi = 150)
    #show_image(io.imread(os.path.join(DIR, img_name)))
    #plt.show()
    #show_sample_images(halo_dataset)

    # Put network on GPU
    #net = neural_net.Net()
    #net = neural_net.StickNet()
    net = neural_net.GeneratorWithAction()

    print(cuda.is_available())
    print(cuda.get_device_name(0))
    gpu = torch.device("cuda:0")
    if USE_GPU:
        net.to(gpu)
    if USE_HALF:
        net.half()

    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.CrossEntropyLoss()
    #criterion2 = nn.BCELoss()
    #criterion = kornia.losses.focal_loss
    #criterion2 = kornia.losses.binary_focal_loss_with_logits
    #criterion2 = nn.BCEWithLogitsLoss()
    #criterion2 = nn.L1Loss()
    #log_var_sticks = torch.zeros((1,), requires_grad=True, device="cuda")
    #log_var_buttons = torch.zeros((1,), requires_grad=True, device="cuda")  
    #params = ([p for p in net.parameters()] + [log_var_sticks] + [log_var_buttons])
    params = net.parameters()
    #optimizer, optimizer_name = optim.SGD(params, lr=LEARNING_RATE), "SGD"
    optimizer, optimizer_name = optim.Adam(params, lr=LEARNING_RATE), "Adam"

    train_set, val_set = torch.utils.data.random_split(halo_dataset, [len(halo_dataset) - VAL_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(SEED))
    print(len(train_set))

    #exp = Experiment("Halo AI")
    exp = Experiment("Halo AI - GeneratorWithAction")
    exp.param("Optimizer", optimizer_name)
    exp.param("learning rate", LEARNING_RATE)
    exp.param("Scheduler", "ReduceLROnPlateau")
    exp.param("loss", "MSE")
    exp.param("Validation Set Size", len(val_set))
    exp.param("Training Set Size", len(train_set))
    exp.param("Batch Size", BATCH_SIZE)
    #train(train_set, val_set, optimizer, criterion, net, exp, save_on_val_perf=True, extra_criterion=criterion2, stick_log_var=log_var_sticks, button_log_var=log_var_buttons)
    train(train_set, val_set, optimizer, criterion, net, exp, save_on_val_perf=True)
    exp.end()
    #net.load(PATH, optimizer, gpu)

    #net.eval()
    #validate(val_set, criterion, net)



    

