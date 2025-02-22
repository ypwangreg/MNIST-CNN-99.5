# Goals
- Create a console version of the C++ CNN for MNIST.
- Show image and results using ascii only [demo, run./test ascii](images/cnn_ascii.png).
- Served as C/C++ programmer playing ground for CNN.
- Expand the program to using multi-cores (CPU) and NVIDIA GPU (RTX3070)
- Explore new ways for the CNN. (current model is slow and chunky. we should have smart ways to do it, like mask, sliding, etc..)

# First Run 2021.12.22
i=1 train=92.68 ent=0.2324,valid=96.05 ent=0.0582 (422sec) lr=1.0e-02 ![firstrun.png](images/firstrun.png)

# Second run 2021.12.27
784x4x10, (785x4+5x10 = 3190) 86.5% in 15secs with 4K parms instead of ~2M parms. [secondrun.png](images/secondrun.png)

# Third run with libvncserver to display change on weights.
Using libvncserver to visualize weights changes
sudo apt install libvncserver-dev


# Ref
1. [MNIST-CNN-99.5] (https://github.com/cdeotte/MNIST-CNN-99.5) Very good and detailed implementation using C. 
2. [libvncserver] (https://packages.ubuntu.com/source/bionic/libvncserver) Used for visualization.
