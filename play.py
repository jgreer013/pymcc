import pyxinput
import torch
import numpy as np
import time
import datetime
import utils
import neural_net
from neural_net import Net
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

PATH = "WGANGP_withPrevAction_NEW_was_mse_30_sgd_16_0_01_e50.pt"

class XController(pyxinput.vController):
    leftJoyX = 'AxisLx'
    leftJoyY = 'AxisLy'
    rightJoyX = 'AxisRx'
    rightJoyY = 'AxisRy'
    leftTrigger = 'TriggerL'
    rightTrigger = 'TriggerR'
    leftBumper = 'BtnShoulderL'
    rightBumper = 'BtnShoulderR'
    a = 'BtnA'
    x = 'BtnX'
    y = 'BtnY'
    b = 'BtnB'
    leftThumb = 'BtnThumbL'
    rightThumb = 'BtnThumbR'
    back = 'BtnBack'
    start = 'BtnStart'
    dpad = 'Dpad'
    current_state = np.random.rand(20) * 2 - 1

    def get_controller_state(self):
        return self.current_state

    def update_controller_state(self, input_vector):
        self.current_state = input_vector
        self.set_value(self.leftJoyX, input_vector[0])
        self.set_value(self.leftJoyY, input_vector[1])
        self.set_value(self.rightJoyX, input_vector[2])
        self.set_value(self.rightJoyY, input_vector[3])
        self.set_value(self.leftTrigger, input_vector[4])
        self.set_value(self.rightTrigger, input_vector[5])
        self.set_value(self.leftBumper, int(input_vector[6]))
        self.set_value(self.rightBumper, int(input_vector[7]))
        self.set_value(self.a, int(input_vector[8]))
        self.set_value(self.x, int(input_vector[9]))
        self.set_value(self.y, int(input_vector[10]))
        self.set_value(self.b, int(input_vector[11]))
        self.set_value(self.leftThumb, int(input_vector[12]))
        self.set_value(self.rightThumb, int(input_vector[13]))
        #self.set_value(self.back, int(input_vector[14]))
        #self.set_value(self.start, int(input_vector[15]))

        dpad = input_vector[16:]
        if dpad[0] == 1:
            self.set_value(self.dpad, self.DPAD_LEFT)
        elif dpad[1] == 1:
            self.set_value(self.dpad, self.DPAD_RIGHT)
        elif dpad[2] == 1:
            self.set_value(self.dpad, self.DPAD_UP)
        elif dpad[3] == 1:
            self.set_value(self.dpad, self.DPAD_DOWN)
        else:
            self.set_value(self.dpad, self.DPAD_OFF)

class ReadController(pyxinput.rController):
    test = "test"

def init_network():
    #net = Net()
    #net = neural_net.StickNet()
    net = neural_net.GeneratorWithAction()

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.device("cuda:0")

    net.load(PATH, gpu=gpu)

    return net

def get_frame_as_tensor():
    sct_img = utils.get_frame()
    sct_img.resize((960, 540))
    np_img = np.asarray(sct_img)
    np_img = np.require(np_img, dtype='f4', requirements=['O', 'W'])
    np_img.setflags(write=True)
    image_tensor = torch.from_numpy(np_img)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor / 255.0
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])
    #image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def get_current_controller_state_as_tensor(controller):
    current_controller_state = torch.from_numpy(controller.get_controller_state())
    current_controller_state = current_controller_state.unsqueeze(0).float()

    return current_controller_state

def fix_action(action):
    action[4:] = (action[4:] + 1) / 2.0
    action[4:] = np.clip(action[4:], 0, 1)
    action[4:] = np.rint(action[4:])

    return action

if __name__ == "__main__":
    controller = XController()
    r_controller = ReadController(2)
    net = init_network()
    net.eval()
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:0')

    print("Plugging in controller in 5 seconds")
    time.sleep(5)
    print("Plugging in controller")
    controller.PlugIn()

    print("Begin playing in 5 seconds (have to give time for controller to be plugged in and to allow you to bring focus to main window")
    time.sleep(5)

    
    while True:
        try:
            input_img = get_frame_as_tensor()

            input_img = input_img.to(gpu)

            current_controller_state = get_current_controller_state_as_tensor(controller)
            current_controller_state = current_controller_state.to(gpu)

            #stick_l_lr, stick_l_ud, stick_r_lr, stick_r_ud, buttons, stick_l_lr_probs, stick_l_ud_probs, stick_r_lr_probs, stick_r_ud_probs, buttons_probs = net(input_img)
            #sticks, buttons, button_probs = net(input_img, current_controller_state)
            #sticks, button_probs = net(input_img, current_controller_state)
            action = net(input_img, current_controller_state)
            #action = net(input_img)
            action = action.to(cpu).detach().numpy()[0]
            #sticks, buttons = torch.split(action, [4, 16], dim=1)
            #buttons = button_probs.to(cpu).detach().numpy()[0]
            #sticks = sticks.to(cpu).detach().numpy()[0]
            #action = np.concatenate((sticks, buttons))
            #sticks, buttons = net(input_img)
            #action = torch.cat((sticks, buttons), 1).to(cpu).detach().numpy()[0]
            action = fix_action(action)
            #print(buttons)
            print(action)
            controller.update_controller_state(action)
        except KeyboardInterrupt:
            break
    print("UnPlugging controller")
    controller.UnPlug()