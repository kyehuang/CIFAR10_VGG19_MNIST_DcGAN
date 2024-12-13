import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from .button_handlers import button_handlers

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 600, 400)

        # Main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Buttons
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_btn)

        self.show_augmented_btn = QPushButton("1. Show Augmented Images")
        self.show_augmented_btn.clicked.connect(button_handlers.show_augmented_images)
        self.layout.addWidget(self.show_augmented_btn)

        self.show_model_structure_btn = QPushButton("2. Show Model Structure")
        self.show_model_structure_btn.clicked.connect(button_handlers.show_model_structure)
        self.layout.addWidget(self.show_model_structure_btn)

        self.show_accuracy_loss_btn = QPushButton("3. Show Accuracy and Loss")
        self.show_accuracy_loss_btn.clicked.connect(button_handlers.show_accuracy_loss)
        self.layout.addWidget(self.show_accuracy_loss_btn)

        self.inference_btn = QPushButton("4. Inference")
        self.inference_btn.clicked.connect(button_handlers.inference)
        self.layout.addWidget(self.inference_btn)

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Prediction label
        self.prediction_label = QLabel("Predicted = ")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_label)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.prediction_label.setText("Predicted = deer")