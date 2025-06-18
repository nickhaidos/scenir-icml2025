import argparse
import json
import logging
import os
import pickle

import metrics
import numpy as np
import torch
from dataset import get_test_dataset, get_train_loader
from model import get_model, get_model_outputs


def train_model(model, train_loader, config, verbose=True):
    # Setup logging - always write to file
    log_file = os.path.join(config['output_path'], 'training_log.txt')
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler - always enabled
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only if verbose
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    discriminator_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=config['learning_rate'])
    encoder_decoder_params = list(model.encoder.parameters()) + list(model.edge_decoder.parameters()) + list(model.feature_decoder.parameters())
    enc_dec_optimizer = torch.optim.AdamW(encoder_decoder_params, lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(enc_dec_optimizer, gamma=config['encoder_lr_scheduler_gamma'])

    model.to(config['device'])

    EPOCHS = config['epochs']
    encoder_loss_log = []
    discriminator_loss_log = []

    model.train()

    for epoch in range(EPOCHS):
        
        running_encoder_loss = 0
        running_discriminator_loss = 0

        for data in train_loader:
            data = data.to(config['device'])
            enc_dec_optimizer.zero_grad()

            z = model.encode(data.x.float(), data.edge_index)

            running_discriminator_loss_temp = 0

            for i in range(5):
                idx = range(data.x.shape[0])  
                model.discriminator.train()
                discriminator_optimizer.zero_grad()
                discriminator_loss = model.discriminator_loss(z[idx]) # Comment
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimizer.step()
                running_discriminator_loss_temp += discriminator_loss.item()

            loss = 0
            loss = loss + (1/6)*model.reg_loss(z)  # Comment

            loss = loss + 3*model.recon_loss_batched(data.x.float(), data.edge_index, data.edge_index, data.batch)
            loss = loss + (1/3)*(1 / data.num_nodes) * model.kl_loss()
            loss.backward()

            enc_dec_optimizer.step()

            running_discriminator_loss += running_discriminator_loss_temp / 5
            running_encoder_loss += loss.item()

        epoch_encoder_loss = running_encoder_loss / len(train_loader)
        encoder_loss_log.append(epoch_encoder_loss)
        scheduler.step()

        epoch_discriminator_loss = running_discriminator_loss / len(train_loader)
        discriminator_loss_log.append(epoch_discriminator_loss)

        epoch_msg = f"Epoch {epoch+1} - Encoder Loss {epoch_encoder_loss:.4f} - Discriminator Loss {epoch_discriminator_loss:.4f}"
        logger.info(epoch_msg)
        
        if (epoch+1) % config['checkpoint_epoch_interval'] == 0:
            # save checkpoint to output path
            checkpoint_path = os.path.join(config['output_path'], f"scenir_ckpt_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_msg = f"Saved model checkpoint for Epoch-{epoch+1}"
            logger.info(checkpoint_msg)

    # Save final model to output path
    final_model_path = os.path.join(config['output_path'], f"scenir_ckpt_epoch_{config['epochs']}.pth")
    torch.save(model.state_dict(), final_model_path)

    return model

def evaluate_output_metrics(config, model_output, k=5):
    try:
        with open(config['test_dataset_ged_path'], "rb") as f:
            test_ged = pickle.load(f)
        print(f"Test data ground-truth loaded from: {config['test_dataset_ged_path']}")
    except Exception as e:
        raise RuntimeError(f"Failed to load test dataset GED file: {str(e)}")

    mask = np.triu_indices(1000, 1)
    test_ged.T[mask[0], mask[1]] = test_ged[mask[0], mask[1]]
    np.fill_diagonal(test_ged, np.nan)     # Same objects don't count
    ged_ground_truth_rankings = np.argsort(test_ged)

    normalized_vecs = model_output / np.expand_dims(np.linalg.norm(model_output, axis=1), -1)
    predicted_similarities = np.matmul(normalized_vecs, normalized_vecs.T)
    np.fill_diagonal(predicted_similarities, np.nan)   # Same objects don't count
    predictions = np.argsort(-predicted_similarities)

    # Handle both single k and iterable of k values
    if isinstance(k, int):
        k_values = [k]
    else:
        k_values = list(k)
    
    # Validate k values are positive integers
    for k_val in k_values:
        if not isinstance(k_val, int) or k_val <= 0:
            raise ValueError(f"All k values must be positive integers, got: {k_val}")

    results = {}
    
    # First 50 ground truth rankings are considered as "correct" answers
    for k_val in k_values:
        map_score, mrr_score, ndcg_score = metrics.compute_metrics(
            ged_ground_truth_rankings[:, :50], predictions, test_ged, "distance", k_val
        )
        results[f"k_{k_val}"] = {
            'map': map_score,
            'mrr': mrr_score,
            'ndcg': ndcg_score
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',                required=True,                        help='path to JSON config file')
    parser.add_argument('--device',                choices=['cpu','cuda'],               help='compute device')
    parser.add_argument('--train',                 action='store_true',                  help='run training phase')
    parser.add_argument('--test',                  action='store_true',                  help='run testing phase')
    parser.add_argument('--train_dataset_path',    help='path to train dataset pickle')
    parser.add_argument('--test_dataset_path',     help='path to test dataset pickle')
    parser.add_argument('--test_dataset_ged_path', help='path to test dataset ground-truth ged values pickle')
    parser.add_argument('--output_path',           required=True,                        help='directory to write metrics/logs/checkpoints')
    parser.add_argument('--verbose',               action='store_true',                  help='enable verbose output')
    parser.add_argument('--checkpoint',            help='path to model checkpoint to load')
    args = parser.parse_args()

    # conditional requirements
    if args.train and not args.train_dataset_path:
        parser.error('--train_dataset_path is required when --train is set')
    if args.test and not args.test_dataset_path:
        parser.error('--test_dataset_path is required when --test is set')
    
    # Ensure output path exists
    try:
        os.makedirs(args.output_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {str(e)}")

    # load and extend config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {str(e)}")
        
    config['device'] = args.device
    config['train_dataset_path'] = args.train_dataset_path
    config['test_dataset_path'] = args.test_dataset_path
    config['test_dataset_ged_path'] = args.test_dataset_ged_path
    config['output_path'] = args.output_path

    # Initialize model before checking train/test
    try:
        model = get_model(config)
        
        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=config['device']))
            print(f"Model loaded from checkpoint: {args.checkpoint}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

    # Train if requested
    if args.train:
        try:
            train_loader = get_train_loader(config)
            if not train_loader:
                raise ValueError("Failed to load training data. Check the dataset path and format.")
            print(f"Training data loaded from: {args.train_dataset_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to prepare training data: {str(e)}")

        try:
            model = train_model(model, train_loader, config, verbose=args.verbose)
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    # Test if requested
    if args.test:
        try:
            test_dataset = get_test_dataset(config)
            if not test_dataset:
                raise ValueError("Failed to load test data. Check the dataset path and format.")
            print(f"Test data loaded from: {args.test_dataset_path}")
            
            model_test_dataset_embeddings = get_model_outputs(model, test_dataset, args.device)

            try:
                output_metrics = evaluate_output_metrics(config, model_test_dataset_embeddings, k=config['retrieval_k_ranks'])
                output_metrics_path = os.path.join(config['output_path'], 'output_metrics.json')
                with open(output_metrics_path, 'w') as f:
                    json.dump(output_metrics, f, indent=4)
                print(f"Output metrics saved to: {output_metrics_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate output metrics: {str(e)}")   

        except Exception as e:
            raise RuntimeError(f"Testing failed: {str(e)}")

if __name__ == '__main__':
    main()