import os
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST, PathMNIST
import sys

# direct file path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # __init__ applied in file A and B
task1_dir = os.path.join(current_dir, 'A')
task2_dir = os.path.join(current_dir, 'B')
sys.path.append(task1_dir)
sys.path.append(task2_dir)


def download_datasets():
    datasets_path = 'Datasets'

    pneumonia_dataset_path = os.path.join(datasets_path, 'PneumoniaMNIST')
    os.makedirs(pneumonia_dataset_path, exist_ok=True)
    PneumoniaMNIST(split='train', download=True, root=pneumonia_dataset_path)
    PneumoniaMNIST(split='val', download=True, root=pneumonia_dataset_path)
    PneumoniaMNIST(split='test', download=True, root=pneumonia_dataset_path)

    pathmnist_dataset_path = os.path.join(datasets_path, 'PathMNIST')
    os.makedirs(pathmnist_dataset_path, exist_ok=True)
    PathMNIST(split='train', download=True, root=pathmnist_dataset_path)
    PathMNIST(split='val', download=True, root=pathmnist_dataset_path)
    PathMNIST(split='test', download=True, root=pathmnist_dataset_path)


def plot_history(history, task_name):
    plt.figure(figsize=(12, 4))

    # Plot the Accuracy VS Epoch curve
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], 'bo', label='Training Accuracy')
    plt.plot(history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'{task_name} Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(0, len(history['accuracy']), 10))  # scale of x-axis is 10
    plt.legend()

    # Plot the Loss VS Epoch curve
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], 'bo', label='Training Loss')
    plt.plot(history['val_loss'], 'r', label='Validation Loss')
    plt.title(f'{task_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history['loss']), 10))  # scale of x-axis is 10
    plt.legend()

    # Save image to the main file
    plt.savefig(f'{task_name}_history.png')
    plt.show()


def main():
    download_datasets()
    from task1_cnn import run as run_task1
    from task2_cnn import run as run_task2

    # create data path for Task A and Task B
    pneumonia_dataset_path = 'Datasets/PneumoniaMNIST'
    pathmnist_dataset_path = 'Datasets/PathMNIST'

    # run task1 and task2
    task1_history, task1_loss, task1_accuracy = run_task1(pneumonia_dataset_path)
    task2_history, task2_loss, task2_accuracy = run_task2(pathmnist_dataset_path)

    print("Task A Results:")
    print(f"Loss: {task1_loss}, Accuracy: {task1_accuracy}")
    plot_history(task1_history, "Task A")

    print("Task B Results:")
    print(f"Loss: {task2_loss}, Accuracy: {task2_accuracy}")
    plot_history(task2_history, "Task B")


if __name__ == "__main__":
    main()
