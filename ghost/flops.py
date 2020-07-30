
from torchstat import stat
from resnet import resnet56
from vgg import vgg16
from ghostrestnet import gresnet56
from ghostvgg import gvgg16



if __name__ == '__main__':

    img_shape = (3, 32, 32)

    resnet56 = resnet56()
    stat(resnet56, img_shape)       # https://github.com/Swall0w/torchstat
    print("↑↑↑↑ is resnet56")
    print("\n"*10)
'''
    ghost_resnet56 = gresnet56()
    stat(ghost_resnet56, img_shape)
    print("↑↑↑↑ is ghost_resnet56")

    
    vgg = 0
    if vgg:
        vgg16 = vgg16()
        stat(vgg16, img_shape)
        print("↑↑↑↑ is vgg16")
        print("\n"*10)

        ghost_vgg16 = gvgg16()
        stat(ghost_vgg16, img_shape)
        print("↑↑↑↑ is ghost_vgg16")
'''
