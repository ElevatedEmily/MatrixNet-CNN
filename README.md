# MatrixNet-CNN
Project Overview
This repository demonstrates MatrixNet, a model that learns to embed discrete image transformations (like rotations, flips, and translations) into invertible matrices, then uses these learned transformation embeddings alongside a CNN for image classification. We use CIFAR-10 as a sample dataset.

Key Features

    Discrete Transformations: We define a small set of transformations (90° rotations, flips, 2px translations).
    Group-Theoretic Embedding: Each generator is mapped to a learnable square matrix, exponentiated to ensure invertibility, and then multiplied in sequence.
    CNN Fusion: The final transformation embedding is concatenated with CNN features from the transformed image for classification.
    Hyperparameter Search: We compare multiple learning rates, matrix sizes, and embedding dimensions.
    Metrics and Plots: We log and plot training/testing loss, accuracy, F1-scores, and confusion matrices.

Dependencies

    Python 3.7+
    PyTorch
    torchvision
    NumPy
    scikit-learn
    matplotlib
    tqdm

File Descriptions

    main.py
        The central script that sets up the dataset, runs training, evaluates performance, and produces plots.
    CIFARTransformed class
        A custom Dataset that applies the random transformations (via apply_transform) to each CIFAR-10 image and encodes a generator “word” for MatrixNet.
    MatrixNet class
        A PyTorch module that maps an encoding of transformations to a matrix representation. It uses torch.matrix_exp to ensure invertibility.
    CNNwithMatrixNet class
        Defines the CNN portion and fuses image features with the MatrixNet embedding for classification.
    Training & Evaluation Functions
        train_epoch and eval_epoch handle the epoch-by-epoch loop with tqdm progress bars.
    Plots
        The script saves line plots of Loss, Accuracy, and F1 vs. Epoch, and also saves a confusion matrix (as .png files).

Running the Project

    Clone or download this repository.
    Install dependencies:

pip install torch torchvision tqdm scikit-learn matplotlib

Run the main script:

        python main.py

By default, it downloads CIFAR-10 into ./data_cifar.
Two hyperparameter configurations are tried. The best result is chosen based on final F1 score.

Outputs
       Plots:
            MatrixNet_Loss_lrXXX_msizeY.png
            MatrixNet_Accuracy_lrXXX_msizeY.png
            MatrixNet_F1_lrXXX_msizeY.png
        Confusion Matrix:
            MatrixNet_CM_lrXXX_msizeY.png
        Console Output:
            Training progress logs, final best hyperparameters, and performance metrics.

  Customization
        Transformations: Add or remove transformations in GENERATOR_SYMBOLS and apply_transform to match your problem.
        Hyperparameters: Modify the param_grid in main to search different matrix sizes, hidden dimensions, or learning rates.
        Epochs: Adjust in run_experiment(..., epochs=...) to train longer or shorter.
        Dataset: Swap out CIFAR10 for another dataset with similar transforms if desired.

  License
    This code is provided as-is under an open-source license (e.g., MIT). Feel free to reuse or modify for research and educational purposes.

  Contact
    For questions or issues, open an Issue or Pull Request in this repository. Contributions and suggestions are welcome!

Enjoy exploring MatrixNet for discrete transformations with CIFAR-10!
