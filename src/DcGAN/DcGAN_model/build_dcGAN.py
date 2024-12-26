from src.DcGAN.DcGAN_model.config import weights_init
from src.DcGAN.DcGAN_model.generator import Generator
from src.DcGAN.DcGAN_model.discriminator import Discriminator

def build_generator(ngpu):
    """
    Build the DcGAN generator model.
    """
    netG = Generator(ngpu)
    netG.apply(weights_init)
    print(netG)
    return netG

def build_discriminator(ngpu):
    """
    Build the DcGAN discriminator model.
    """
    netD = Discriminator(ngpu)
    netD.apply(weights_init)
    print(netD)
    return netD

if __name__ == "__main__":
    build_generator(1)
    build_discriminator(1)
