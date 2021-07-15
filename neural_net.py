import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, padding = 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 48, 5, padding = 2)
        self.bn4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 64, 5, padding = 2)
        self.bn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 60 * 33, 240)
        self.bnf1 = nn.BatchNorm1d(240)
        self.fc2 = nn.Linear(240, 120)
        self.bnf2 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 50)
        self.bnf3 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x): # 1920 x 1080 x 3
        x = self.bn1(self.pool(self.relu(self.conv1(x)))) # 960 x 540
        x = self.bn2(self.pool(self.relu(self.conv2(x)))) # 480 x 270
        x = self.bn3(self.pool(self.relu(self.conv3(x)))) # 240 x 135
        x = self.bn4(self.pool(self.relu(self.conv4(x)))) # 120 x 67
        x = self.bn5(self.pool(self.relu(self.conv5(x)))) # 60 x 33
        x = x.view(-1, 64 * 60 * 33)
        x = self.bnf1(self.relu(self.fc1(x)))
        x = self.bnf2(self.relu(self.fc2(x)))
        x = self.bnf3(self.relu(self.fc3(x)))
        x = self.tanh(self.fc4(x))
        return x

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class StickNet(nn.Module):
    def __init__(self):
        super(StickNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, padding = 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 48, 5, padding = 2)
        self.bn4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 64, 5, padding = 2)
        self.bn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 60 * 33, 240)
        self.bnf1 = nn.BatchNorm1d(240)
        self.fc2 = nn.Linear(240, 240)
        self.bnf2 = nn.BatchNorm1d(240)
        self.fc3 = nn.Linear(240, 240)
        self.bnf3 = nn.BatchNorm1d(240)
        self.fc4 = nn.Linear(240, 4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x): # 1920 x 1080 x 3
        x = self.bn1(self.pool(self.relu(self.conv1(x)))) # 960 x 540
        x = self.bn2(self.pool(self.relu(self.conv2(x)))) # 480 x 270
        x = self.bn3(self.pool(self.relu(self.conv3(x)))) # 240 x 135
        x = self.bn4(self.pool(self.relu(self.conv4(x)))) # 120 x 67
        x = self.bn5(self.pool(self.relu(self.conv5(x)))) # 60 x 33
        x = x.view(-1, 64 * 60 * 33)
        x = self.bnf1(self.relu(self.fc1(x)))
        x = self.bnf2(self.relu(self.fc2(x)))
        x = self.bnf3(self.relu(self.fc3(x)))
        x = self.tanh(self.fc4(x))
        return x

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class ModifiedResnet(nn.Module):
    def __init__(self):
        super(ModifiedResnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1000)
        self.bnf = nn.BatchNorm1d(1000)
        self.fcf = nn.Linear(1000, 20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.resnet18(x))
        x = self.tanh(self.fcf(x))
        return x

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class MixedActivationResnet(nn.Module):
    def __init__(self):
        super(MixedActivationResnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_stick = nn.Linear(512, 64)
        self.bn_stick = nn.BatchNorm1d(64)
        self.fcf_stick = nn.Linear(64, 4)

        self.fc_button = nn.Linear(512, 64)
        self.bn_button = nn.BatchNorm1d(64)
        self.fcf_button = nn.Linear(64, 16)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.resnet18(x))
        sticks = self.bn_stick(self.relu(self.fc_stick(x)))
        sticks = self.tanh(self.fcf_stick(sticks))
        buttons = self.bn_button(self.relu(self.fc_button(x)))
        buttons = self.sig(self.fcf_button(buttons))
        return sticks, buttons

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class MixedActivationResnet_ActionGenerator(nn.Module):
    def __init__(self):
        super(MixedActivationResnet_ActionGenerator, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_stick = nn.Linear(512, 64)
        self.bn_stick = nn.BatchNorm1d(64)
        self.fcf_stick = nn.Linear(64, 4)

        self.fc_button = nn.Linear(512, 64)
        self.bn_button = nn.BatchNorm1d(64)
        self.fcf_button = nn.Linear(64, 16)
        
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.resnet18(x))
        sticks = self.bn_stick(self.relu(self.fc_stick(x)))
        sticks = self.tanh(self.fcf_stick(sticks))
        button_probs = self.bn_button(self.relu(self.fc_button(x)))
        button_probs = self.sig(self.fcf_button(button_probs))
        return sticks, button_probs

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class GeneratorWithActionTanh(nn.Module):
    def __init__(self):
        super(GeneratorWithActionTanh, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_action = nn.Linear(20, 64)
        self.fc_state = nn.Linear(576, 512)
        self.bn_state = nn.BatchNorm1d(512)

        self.fc_stick = nn.Linear(512, 64)
        self.bn_stick = nn.BatchNorm1d(64)
        self.fcf_stick = nn.Linear(64, 4)

        self.fc_button = nn.Linear(512, 64)
        self.bn_button = nn.BatchNorm1d(64)
        self.fcf_button = nn.Linear(64, 16)
        
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, image, action):
        image = self.relu(self.resnet18(image))
        action = self.relu(self.fc_action(action))
        state = self.bn_state(self.relu(self.fc_state(torch.cat((image, action), dim=1))))
        sticks = self.bn_stick(self.relu(self.fc_stick(state)))
        sticks = self.tanh(self.fcf_stick(sticks))
        button_tanh = self.bn_button(self.relu(self.fc_button(state)))
        button_tanh = self.tanh(self.fcf_button(button_tanh))
        return sticks, button_tanh

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class GeneratorWithAction(nn.Module):
    def __init__(self):
        super(GeneratorWithAction, self).__init__()
        n_hidden = 1000
        n_action = 100
        n_sum = n_hidden + n_action
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, n_hidden)
        self.bn_resnet = nn.BatchNorm1d(n_hidden)

        self.fc_action = nn.Linear(20, n_action)
        self.bn_action = nn.BatchNorm1d(n_action)

        self.fc_state = nn.Linear(n_sum, n_hidden)
        self.bn_state = nn.BatchNorm1d(n_hidden)

        self.fc_final_state = nn.Linear(n_hidden, n_hidden)
        self.bn_final_state = nn.BatchNorm1d(n_hidden)

        self.fc_final = nn.Linear(n_hidden, 20)
        self.bn_final = nn.BatchNorm1d(20)
        
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, image, action):
        image = self.relu(self.bn_resnet(self.resnet18(image)))
        action = self.relu(self.bn_action(self.fc_action(action)))
        state = self.relu(self.bn_state(self.fc_state(torch.cat((image, action), dim=1))))
        state = self.relu(self.bn_final_state(self.fc_final_state(state)))
        generated_action = self.tanh(self.bn_final(self.fc_final(state)))
        return generated_action

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class ResnetImageActionDiscriminator(nn.Module):
    def __init__(self):
        super(ResnetImageActionDiscriminator, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_action = nn.Linear(20, 64)

        # Combine action tensor with image tensor - 64 + 512
        self.bn_concat = nn.BatchNorm1d(576)
        self.fc_concat = nn.Linear(576, 512)
        self.bn_reduced = nn.BatchNorm1d(512)

        self.fc_combined = nn.Linear(512, 64)
        self.fc_bn = nn.BatchNorm1d(64)
        self.fc_final = nn.Linear(64, 1)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, image, action):
        image = self.relu(self.resnet18(image))
        action = self.relu(self.fc_action(action)) # 64
        concat = self.bn_concat(torch.cat((image, action), dim=1)) # 576
        concat = self.bn_reduced(self.relu(self.fc_concat(concat))) # 512
        concat = self.fc_bn(self.relu(self.fc_combined(concat)))
        prob = self.sig(self.fc_final(concat))
        return prob

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class ResnetImageActionDiscriminatorWGAN(nn.Module):
    def __init__(self):
        super(ResnetImageActionDiscriminatorWGAN, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_action = nn.Linear(20, 64)

        # Combine action tensor with image tensor - 64 + 512
        self.bn_concat = nn.BatchNorm1d(576)
        self.fc_concat = nn.Linear(576, 512)
        self.bn_reduced = nn.BatchNorm1d(512)

        self.fc_combined = nn.Linear(512, 64)
        self.fc_bn = nn.BatchNorm1d(64)
        self.fc_final = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, image, action):
        image = self.relu(self.resnet18(image))
        action = self.relu(self.fc_action(action)) # 64
        concat = self.bn_concat(torch.cat((image, action), dim=1)) # 576
        concat = self.bn_reduced(self.relu(self.fc_concat(concat))) # 512
        concat = self.fc_bn(self.relu(self.fc_combined(concat)))
        prob = self.fc_final(concat)
        return prob

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class ResnetImageActionDiscriminatorWGANGPWithAction(nn.Module):
    def __init__(self):
        super(ResnetImageActionDiscriminatorWGANGPWithAction, self).__init__()
        n_hidden = 1000
        n_action = 20
        n_sum = n_hidden + n_action
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, n_hidden)
        #self.ln_resnet = nn.LayerNorm(n_hidden)

        self.fc_action = nn.Linear(20, n_action)
        #self.ln_action = nn.LayerNorm(n_action)

        self.fc_prev_action = nn.Linear(20, n_action)
        #self.ln_prev_action = nn.LayerNorm(n_action)

        self.fc_state = nn.Linear(n_sum, n_hidden)
        #self.ln_state = nn.LayerNorm(n_hidden)
        # Combine action tensor with image tensor - 64 + 512
        self.fc_concat = nn.Linear(n_sum, n_hidden)
        #self.ln_reduced = nn.LayerNorm(n_hidden)

        self.fc_combined = nn.Linear(n_hidden, n_hidden)
        #self.ln_combined = nn.LayerNorm(n_hidden)
        self.fc_final = nn.Linear(n_hidden, 1)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, image, action, prev_action):
        #image = self.relu(self.ln_resnet(self.resnet18(image)))
        image = self.relu(self.resnet18(image))
        #action = self.relu(self.ln_action(self.fc_action(action))) # 64
        action = self.relu(self.fc_action(action))
        #prev_action = self.relu(self.ln_prev_action(self.fc_prev_action(prev_action))) # 64
        prev_action = self.relu(self.fc_prev_action(prev_action))
        #state = self.relu(self.ln_state(self.fc_state(torch.cat((image, prev_action), dim=1)))) # 512
        state = self.relu(self.fc_state(torch.cat((image, prev_action), dim=1)))
        concat = torch.cat((state, action), dim=1) # 576
        #concat = self.relu(self.ln_reduced(self.fc_concat(concat))) # 512
        concat = self.relu(self.fc_concat(concat))
        #concat = self.relu(self.ln_combined(self.fc_combined(concat))) # 64
        concat = self.relu(self.fc_combined(concat))
        prob = self.fc_final(concat) # 1
        return prob

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class ResnetImageActionDiscriminatorWGANGP(nn.Module):
    def __init__(self):
        super(ResnetImageActionDiscriminatorWGANGP, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_action = nn.Linear(20, 64)

        # Combine action tensor with image tensor - 64 + 512
        self.ln_concat = nn.LayerNorm(576)
        self.fc_concat = nn.Linear(576, 512)
        self.ln_reduced = nn.LayerNorm(512)

        self.fc_combined = nn.Linear(512, 64)
        self.ln_combined = nn.LayerNorm(64)
        self.fc_final = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, image, action):
        image = self.relu(self.resnet18(image))
        action = self.relu(self.fc_action(action)) # 64
        concat = self.ln_concat(torch.cat((image, action), dim=1)) # 576
        concat = self.ln_reduced(self.relu(self.fc_concat(concat))) # 512
        concat = self.ln_combined(self.relu(self.fc_combined(concat))) # 64
        prob = self.fc_final(concat) # 1
        return prob

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class MixedActivationResnetWithActionAndButtonLogits(nn.Module):
    def __init__(self):
        super(MixedActivationResnetWithActionAndButtonLogits, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_stick = nn.Linear(576, 64)
        self.bn_stick = nn.BatchNorm1d(64)
        self.fcf_stick = nn.Linear(64, 4)

        self.fc_button = nn.Linear(576, 64)
        self.bn_button = nn.BatchNorm1d(64)
        self.fcf_button = nn.Linear(64, 16)

        self.fc_action = nn.Linear(20, 64)

        # Combine action tensor with image tensor - 64 + 512
        self.bn_concat = nn.BatchNorm1d(576)
        self.fc_concat = nn.Linear(576, 576)
        self.bn_reduced = nn.BatchNorm1d(576)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, action):
        x = self.relu(self.resnet18(x))
        action = self.relu(self.fc_action(action)) # 64
        concat = self.bn_concat(torch.cat((x, action), dim=1)) # 576
        concat = self.bn_reduced(self.relu(self.fc_concat(concat))) # 512
        sticks = self.bn_stick(self.relu(self.fc_stick(concat)))
        sticks = self.tanh(self.fcf_stick(sticks))
        buttons = self.bn_button(self.relu(self.fc_button(concat)))
        buttons = self.fcf_button(buttons)
        button_probs = self.sig(buttons)
        return sticks, buttons, button_probs

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class Resnet34WithPreviousAction(nn.Module):
    def __init__(self):
        super(Resnet34WithPreviousAction, self).__init__()
        n_hidden = 2000
        n_hidden_action = 64
        self.resnet34 = models.resnet34(pretrained=False)
        num_features = self.resnet34.fc.in_features
        print(num_features)
        self.resnet34.fc = nn.Linear(num_features, n_hidden)
        n_sum = n_hidden + n_hidden_action

        self.fc_stick = nn.Linear(n_sum, n_hidden)
        self.bn_stick = nn.BatchNorm1d(n_hidden)
        self.fc_final = nn.Linear(n_hidden, 20)

        self.fc_action = nn.Linear(20, n_hidden_action)

        # Combine action tensor with image tensor - 64 + 512
        #self.bn_concat = nn.BatchNorm1d(n_sum)
        self.fc_concat = nn.Linear(n_sum, n_sum)
        self.bn_reduced = nn.BatchNorm1d(n_sum)
        
        self.relu = nn.ReLU()
        #self.relu = nn.Tanh()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, action):
        x = self.relu(self.resnet34(x))
        action = self.relu(self.fc_action(action)) # 64
        concat = torch.cat((x, action), dim=1) # 576
        concat = self.relu(self.bn_reduced(self.fc_concat(concat))) # 576
        output_action = self.relu(self.bn_stick(self.fc_stick(concat))) # 64
        output_action = self.tanh(self.fc_final(output_action)) # 20
        return output_action

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        n_hidden = 2000
        self.resnet34 = models.resnet34(pretrained=False)
        num_features = self.resnet34.fc.in_features
        print(num_features)
        self.resnet34.fc = nn.Linear(num_features, n_hidden)

        self.bn_resnet = nn.BatchNorm1d(n_hidden)
        self.fc_stick = nn.Linear(n_hidden, n_hidden)
        self.bn_stick = nn.BatchNorm1d(n_hidden)
        self.fc_final = nn.Linear(n_hidden, 20)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn_resnet(self.resnet34(x)))
        output_action = self.relu(self.bn_stick(self.fc_stick(x)))
        output_action = self.tanh(self.fc_final(output_action)) # 20
        return output_action

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class MixedActivationClassificationResnet(nn.Module):
    def __init__(self):
        super(MixedActivationClassificationResnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_stick = nn.Linear(512, 32)
        self.bn_stick = nn.BatchNorm1d(32)
        self.fcf_stick_l_lr = nn.Linear(32, 5)
        self.fcf_stick_l_ud = nn.Linear(32, 5)
        self.fcf_stick_r_lr = nn.Linear(32, 5)
        self.fcf_stick_r_ud = nn.Linear(32, 5)

        self.fc_button = nn.Linear(512, 64)
        self.bn_button = nn.BatchNorm1d(64)
        self.fcf_button = nn.Linear(64, 16)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.resnet18(x))

        # sticks converted to multi-class problem with 5 classes each, to be converted to one of [-1, -0.5, 0, 0.5, 1]
        sticks = self.bn_stick(self.relu(self.fc_stick(x)))

        # CrossEntropy applies softmax by itself, so no need to pass it here
        stick_l_lr = self.fcf_stick_l_lr(sticks)
        stick_l_ud = self.fcf_stick_l_ud(sticks)
        stick_r_lr = self.fcf_stick_r_lr(sticks)
        stick_r_ud = self.fcf_stick_r_ud(sticks)

        # Output these to determine class for output at runtime
        stick_l_lr_probs = self.softmax(stick_l_lr)
        stick_l_ud_probs = self.softmax(stick_l_ud)
        stick_r_lr_probs = self.softmax(stick_r_lr)
        stick_r_ud_probs = self.softmax(stick_r_ud)

        buttons = self.bn_button(self.relu(self.fc_button(x)))
        buttons = self.sig(self.fcf_button(buttons))
        return stick_l_lr, stick_l_ud, stick_r_lr, stick_r_ud, buttons, stick_l_lr_probs, stick_l_ud_probs, stick_r_lr_probs, stick_r_ud_probs

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class MixedActivationFocalClassificationResnet(nn.Module):
    def __init__(self):
        super(MixedActivationFocalClassificationResnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 512)

        self.fc_stick = nn.Linear(512, 32)
        self.bn_stick = nn.BatchNorm1d(32)
        self.fcf_stick_l_lr = nn.Linear(32, 5)
        self.fcf_stick_l_ud = nn.Linear(32, 5)
        self.fcf_stick_r_lr = nn.Linear(32, 5)
        self.fcf_stick_r_ud = nn.Linear(32, 5)

        self.fc_button = nn.Linear(512, 64)
        self.bn_button = nn.BatchNorm1d(64)
        self.fcf_button = nn.Linear(64, 16)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.resnet18(x))

        # sticks converted to multi-class problem with 5 classes each, to be converted to one of [-1, -0.5, 0, 0.5, 1]
        sticks = self.bn_stick(self.relu(self.fc_stick(x)))

        # CrossEntropy applies softmax by itself, so no need to pass it here
        stick_l_lr = self.fcf_stick_l_lr(sticks)
        stick_l_ud = self.fcf_stick_l_ud(sticks)
        stick_r_lr = self.fcf_stick_r_lr(sticks)
        stick_r_ud = self.fcf_stick_r_ud(sticks)

        # Output these to determine class for output at runtime
        stick_l_lr_probs = self.softmax(stick_l_lr)
        stick_l_ud_probs = self.softmax(stick_l_ud)
        stick_r_lr_probs = self.softmax(stick_r_lr)
        stick_r_ud_probs = self.softmax(stick_r_ud)

        buttons = self.bn_button(self.relu(self.fc_button(x)))
        # Focal BCE uses logits, so we don't want to apply this to the main output
        buttons = self.fcf_button(buttons)
        buttons_probs = self.sig(buttons)
        return stick_l_lr, stick_l_ud, stick_r_lr, stick_r_ud, buttons, stick_l_lr_probs, stick_l_ud_probs, stick_r_lr_probs, stick_r_ud_probs, buttons_probs

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)

class NetMixedActivation(nn.Module):
    def __init__(self):
        super(NetMixedActivation, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding = 2)
        self.conv4 = nn.Conv2d(32, 48, 5, padding = 2)
        self.conv5 = nn.Conv2d(48, 64, 5, padding = 2)
        self.fc1 = nn.Linear(64 * 60 * 33, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 50)
        self.fc4 = nn.Linear(50, 20)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x): # 1920 x 1080 x 3
        x = self.pool(self.relu(self.conv1(x))) # 960 x 540
        x = self.pool(self.relu(self.conv2(x))) # 480 x 270
        x = self.pool(self.relu(self.conv3(x))) # 240 x 135
        x = self.pool(self.relu(self.conv4(x))) # 120 x 67
        x = self.pool(self.relu(self.conv5(x))) # 60 x 33
        x = x.view(-1, 64 * 60 * 33)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        split = torch.split(x, [4, 16], 1) # split tri
        joysticks = self.tanh(split[0])
        other_buttons = self.sig(split[1])

        return torch.cat((joysticks, other_buttons), 1)

    def load(self, path, optimizer=None, gpu=None):
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if gpu:
            torch.cuda.empty_cache()
            self.to(gpu)
