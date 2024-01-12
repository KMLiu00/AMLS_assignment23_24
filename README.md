AMLS Assignment Code


This project contains code for the AMLS (Applied Machine Learning Systems) assignment. It includes two main tasks focused on different datasets, PneumoniaMNIST and PathMNIST, each with their respective visualization components.

Structure
- `main.py`: The main script to run the tasks. It orchestrates the execution of both tasks and manages dataset downloads.
- `Datasets`: Directory where datasets are downloaded and stored.

Task A (PneumoniaMNIST Analysis)
- Located in the `A` directory.
- `task1.py`: Contains the model and data processing for the PneumoniaMNIST dataset.
- `task1_visualisation.py`: Provides visualizations for the Task A's train dataset and print Task A datastructure.

Task B (PathMNIST Analysis)
- Located in the `B` directory.
- `task2.py`: Contains the model and data processing for the PathMNIST dataset.
- `task2_visualisation.py`: Provides visualizations for the Task B's train dataset and print Task B datastructure.
---
Usage

To run the entire project:
1. Navigate to the project directory.
2. Run `python main.py`.

This will download the required datasets into the `Datasets` folder and execute the analysis for both tasks.

Dependencies
- TensorFlow
- NumPy
- Matplotlib
- MedMNIST (for data loading)

Visualization

1.After running `main.py`, it will generate plots to visually interpret the model's performance and data analysis results.

2.The visualization scripts in `A/task1_visualisation.py` will generate plots to visually display the first ten images of the PneumoniaMNIST dataset. The visualization scripts `B/task2_visualisation.py` will generate plots to visually diplay the first ten images of each category (9 categories in total). (cannot run by main., run individually)


