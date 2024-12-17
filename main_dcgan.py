"""
This is the main file for the DcGAN project.
"""
import sys
from PyQt5.QtWidgets import QApplication

from src.DcGAN.gui.main_window import MainWindow


def main():
    """
    Main function for the CIFAR10_VGG19 project.
    """
    app = QApplication(sys.argv) # app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
