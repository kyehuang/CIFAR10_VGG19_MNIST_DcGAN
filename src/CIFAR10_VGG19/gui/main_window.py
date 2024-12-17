"""
Main window of the application about CIFAR10_VGG19
"""
import dataclasses
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from src.CIFAR10_VGG19.gui.button_handlers import ButtonHandlers

@dataclasses.dataclass
class MainWindow(QMainWindow):
    """
    Main window of the application about CIFAR10_VGG19
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

        # # Image layout
        # # Image display
        # self.image_label = QLabel(self)
        # self.image_label.setAlignment(Qt.AlignCenter)
        # self.layout.addWidget(self.image_label)

        # # Prediction label
        # self.prediction_label = QLabel("Predicted = ")
        # self.prediction_label.setAlignment(Qt.AlignCenter)
        # self.layout.addWidget(self.prediction_label)

    def load_image(self):
        """
        Load image from file
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.prediction_label.setText("Predicted = deer")

@dataclasses.dataclass
class ButtonPanel(QWidget):
    """
    Panel with buttons
    """
    def __init__(self, parent, button_handler: ButtonHandlers):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.button_handler = button_handler
        self.parent = parent
        self.image_path = None

        # Add buttons
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.call_load_image)
        self.layout.addWidget(self.load_image_btn)

        self.show_augmented_btn = QPushButton("1. Show Augmented Images")
        self.show_augmented_btn.clicked.connect(button_handler.show_augmented_images)
        self.layout.addWidget(self.show_augmented_btn)

        self.show_model_structure_btn = QPushButton("2. Show Model Structure")
        self.show_model_structure_btn.clicked.connect(button_handler.show_model_structure)
        self.layout.addWidget(self.show_model_structure_btn)

        self.show_accuracy_loss_btn = QPushButton("3. Show Accuracy and Loss")
        self.show_accuracy_loss_btn.clicked.connect(button_handler.show_accuracy_loss)
        self.layout.addWidget(self.show_accuracy_loss_btn)

        self.inference_btn = QPushButton("4. Inference")
        self.inference_btn.clicked.connect(self.call_inference)
        self.layout.addWidget(self.inference_btn)

                # Image layout
        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Prediction label
        self.prediction_label = QLabel("Predicted = ")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_label)

    def call_inference(self):
        """
        Call inference
        """
        result = self.button_handler.do_inference(self, self.image_path)

        print(result)

    def call_load_image(self):
        """
        Load image from file
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        self.image_path = file_name
    