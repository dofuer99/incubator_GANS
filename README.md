## Incubator_GANS

You don't need to down load extra files to run the code.

'gan_week_3.py' is the basic version.

'gan_cat.py' is the version where several person are grouped as one input sample. One sample is represented by a vector.
Traing will fail if the number of person of one sample is higher than 30.

'gan_data_conv.py' is the version where several person are grouped as one input sample and one sampled is represented by an image.
Convolutional NN might be nice here because both the input and output are images. However, due to the simplicity, Fully connected NN
still works here.
