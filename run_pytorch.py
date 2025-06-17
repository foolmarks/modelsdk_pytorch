import argparse
import os
import logging
import numpy as np
import torch


def load_model(path):
    """
    Load a fully trained PyTorch model from the given file path.
    """
    try:
        model = torch.load(path)
    except RuntimeError:
        raise RuntimeError(
            "Failed to load full model. Please save whole model with torch.save(model)"
        )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained PyTorch ImageNet model on data from a NumPy .npz file.'
    )

    parser.add_argument(
        '--model_path', '-m', type=str,
        default='./pyt/resnext101_32x8d_wsl.pt',
        help='Path to the trained PyTorch model file (.pt or .pth)'
    )
    parser.add_argument(
        '--npz_path', '-n', type=str,
        default='./test_data.npz',
        help="Path to the .npz file containing 'x' (inputs) and 'y' (labels)"
    )

    args = parser.parse_args()

    # Derive log file name from model_path
    model_base = os.path.splitext(os.path.basename(args.model_path))[0]
    log_file = "run_pytorch.log"

    # Configure logging to file
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path).to(device)
    logging.info("Model loaded and set to evaluation mode")

    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    # Load data
    logging.info("Loading data from %s", args.npz_path)
    data = np.load(args.npz_path)
    x = data['x']  # expected shape: (N, H, W, C)
    y = data['y']  # expected shape: (N,)

    total = x.shape[0]
    correct = 0
    logging.info("Starting evaluation on %d samples", total)

    for i in range(total):
        img_array = x[i]  # shape: (H, W, C), values 0-255

        # Preprocess: convert to [0,1], permute to C,H,W, normalize
        tensor = torch.from_numpy(img_array).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - mean) / std
        input_tensor = tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)

        # Get predicted class index
        pred_idx = torch.argmax(outputs, dim=1).item()

        # Compare with ground truth label
        true_label = y[i]
        if pred_idx == true_label:
            correct += 1

        # Log prediction vs label
        logging.info("Sample %d: predicted %d, label %d", i, pred_idx, true_label)

    accuracy = (correct / total) * 100
    logging.info("Evaluation complete: %d correct out of %d (%.2f%%)", correct, total, accuracy)

    # Also print a summary to console
    print(f'Evaluated {total} samples')
    print(f'Accuracy: {accuracy:.2f}% ({correct}/{total})')


if __name__ == '__main__':
    main()
