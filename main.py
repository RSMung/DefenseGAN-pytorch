

# DefenseGAN
from defense_gan import GradientDescentReconstruct

attack_type = "FGSM"
eps = 0.01

gd = GradientDescentReconstruct()
gd.reconstruct(attack_type, eps)