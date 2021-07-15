import mss
import cv2
import numpy as np

class FrameData():

    def __init__(self, fileName: str, controllerState: list, imageData):
        self.fileName = fileName
        self.controllerState = controllerState
        self.imageData = imageData

    def save(self, outfile):
        self.imageData.save(self.fileName)

        # write csv line
        outfile.write( self.fileName + ',' + ','.join(map(str, self.controllerState)) + '\n' )

