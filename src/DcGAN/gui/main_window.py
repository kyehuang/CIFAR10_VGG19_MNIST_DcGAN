"""
Main window of the application about DcGAN
"""
import dataclasses
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from src.DcGAN.gui.button_handlers import ButtonHandlers

@dataclasses.dataclass
class MainWindow(QMainWindow):
    """
    Main window of the application about DcGAN
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MainWindow") # Set window title
        self.setGeometry(100, 100, 600, 400) # Set window geometry

        # Main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Button handlers
        self.button_handlers = ButtonHandlers(self)

        # Buttons layout
        self.button_panel = ButtonPanel(self, self.button_handlers)
        self.layout.addWidget(self.button_panel)


@dataclasses.dataclass
class ButtonPanel(QWidget):
    """
    Panel with buttons
    """
    def __init__(self, parent, button_handler):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add buttons

        self.show_training_images_btn = QPushButton("1. Show Training Images")
        self.show_training_images_btn.clicked.connect(button_handler.show_training_images)
        self.layout.addWidget(self.show_training_images_btn)

        self.show_model_structure_btn = QPushButton("2. Show Model Structure")
        self.show_model_structure_btn.clicked.connect(button_handler.show_model_structure)
        self.layout.addWidget(self.show_model_structure_btn)

        self.show_training_loss_btn = QPushButton("3. Show Training Loss")
        self.show_training_loss_btn.clicked.connect(button_handler.show_training_loss)
        self.layout.addWidget(self.show_training_loss_btn)

        self.inference_btn = QPushButton("4. Inference")
        self.inference_btn.clicked.connect(button_handler.inference)
        self.layout.addWidget(self.inference_btn)
