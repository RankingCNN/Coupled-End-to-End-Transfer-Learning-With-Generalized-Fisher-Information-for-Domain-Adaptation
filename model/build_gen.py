from .svhn2mnist import Feature
from .svhn2mnist import Predictor
from .svhn2mnist import Decoder
#import usps
#import syn2gtrsb
#import syndig2svhn

def Generator(source, target, pixelda=False):    
    if source == 'svhn':
        return Feature()
    


def Classifier(source, target):    
    if source == 'svhn':
        return Predictor()
    
        
        
def D(source, target):
    if source=='svhn':
        return Decoder()
