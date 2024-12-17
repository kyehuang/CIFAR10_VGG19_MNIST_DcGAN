"""
Time: 2021-07-25
Author: Kye Huang
"""
import matplotlib.pyplot as plt

def show_predicted_distribution(data: dict) -> None:
    """
    Show the predicted distribution of classes

    Args:
    data (dict): dictionary containing class labels and their probabilities

    Returns:
    None
    """

    # Extract data
    labels = list(data.keys())
    scores = list(data.values())

    # Create bar plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, scores, color='royalblue')

    # Highlight the "horse" as the most likely class
    for bar_iter, score in zip(bars, scores):
        plt.text(bar_iter.get_x() + bar_iter.get_width()/2, bar_iter.get_height(),
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Add titles and labels
    plt.title("Probability of each class", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Probability", fontsize=12)

    # Adjust layout for readability
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_DATA = {'plane': 6.104882572799397e-07,
            'automobile': 3.275400928259842e-08, 
            'bird': 8.899998249489727e-08, 
            'cat': 3.89324945615499e-08, 
            'deer': 2.8323327683210664e-07, 
            'dog': 1.1689164693962084e-06, 
            'frog': 2.1120143134378822e-10, 
            'horse': 0.9999971389770508, 
            'ship': 3.874569831641139e-10, 
            'truck': 7.581602972095425e-07}
    show_predicted_distribution(TEST_DATA)
