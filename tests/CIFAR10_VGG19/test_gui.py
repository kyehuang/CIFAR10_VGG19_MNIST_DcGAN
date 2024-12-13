import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap
from src.CIFAR10_VGG19.gui.main_window import MainWindow  # 假設主程式文件名為 main.py

class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # init QApplication
        cls.app = QApplication([])

    @classmethod
    def tearDownClass(cls):
        # 關閉應用程序
        cls.app.quit()

    def setUp(self):
        # init MainWindow
        self.window = MainWindow()
        self.window.show()  # 確保窗口已調用 show() 來顯示它

    def test_main_window_loads(self):
        """
        test if the main window loads
        """
        self.assertIsNotNone(self.window)
        self.assertTrue(self.window.isVisible())

    def test_buttons_exist(self):
        """測試按鈕是否存在"""
        self.assertIsNotNone(self.window.load_image_btn)
        self.assertIsNotNone(self.window.show_augmented_btn)
        self.assertIsNotNone(self.window.show_model_structure_btn)
        self.assertIsNotNone(self.window.show_accuracy_loss_btn)
        self.assertIsNotNone(self.window.inference_btn)

    def test_image_label_initial_state(self):
        """測試圖像標籤初始狀態"""
        self.assertEqual(self.window.image_label.pixmap(), None)

    def test_load_image(self):
        """測試加載圖像功能"""
        # 模擬加載圖像
        self.window.load_image()
        # 測試是否顯示了圖像（假設測試時提供有效文件路徑）
        pixmap = QPixmap("dataset/Q1_image/Q1_1/automobile.png")  # 測試圖像路徑
        self.window.image_label.setPixmap(pixmap)
        self.assertIsNotNone(self.window.image_label.pixmap())
        self.assertEqual(self.window.prediction_label.text(), "Predicted = deer")


if __name__ == "__main__":
    unittest.main()
