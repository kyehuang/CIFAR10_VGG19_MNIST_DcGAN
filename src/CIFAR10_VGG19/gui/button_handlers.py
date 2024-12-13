class button_handlers:
    def __init__(self):
        pass

    def show_augmented_images(self):
        try:
            print("Show Augmented Images")
            return True
        except Exception as e:
            print(e)
            return False

    def show_model_structure(self):
        try:
            print("Show Model Structure")
            return True
        except Exception as e:
            print(e)
            return False

    def show_accuracy_loss(self):
        try:
            print("Show Accuracy Loss")
            return True
        except Exception as e:
            print(e)
            return False

    def inference(self):
        try:
            print("Inference")
            return True
        except Exception as e:
            print(e)
            return False
