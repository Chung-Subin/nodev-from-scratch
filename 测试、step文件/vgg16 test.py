import nodev
from PIL import Image
import numpy as np
img_path= '../img.png'
img=Image.open(img_path)
#img.show()
from nodev.model import VGG16
x=VGG16.preprocess(img)
x=x[np.newaxis]
model=VGG16(pretrained=True)
with nodev.test_mode():
    y=model(x)
    pre_id=np.argmax(y.data)
#model.plot(x,to_file='vgg.pdf')
labels=nodev.dataset.ImageNet.labels()
print(labels[pre_id])

