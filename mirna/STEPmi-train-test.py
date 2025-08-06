import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr,spearmanr
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model.models import STEPmi
import argparse
import random
from scipy.spatial.distance import cdist
from mirna.create_graph_and_embeddings_STEPmi import TemporalGraphDatasetMirna
from step_losses import miRNA_enhanced_temporal_loss
from sklearn.manifold import TSNE


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

def process_batch(seq, label):

    # Input --> temporal node2vec embeddings
    x = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features] --> [10, 162 (num of genes), 256 (embeddings)]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, features, seq_len, nodes]
    
    # Target: only expression values for comparison
    target = torch.stack([g.x[:, -1] for g in label])  # [1, nodes] (expression values)
    target = target.unsqueeze(1).unsqueeze(0)  # [1, 1, 1, nodes]
    
    return x, target

def compute_gene_connections(dataset):
    """
    Function for model STEPmi, calculation of gene-attention pairs
    """
    connections = {}
    for idx, gene in enumerate(dataset.node_map.keys()):
        count1 = len(dataset.df[dataset.df['Gene1'] == gene])
        count2 = len(dataset.df[dataset.df['Gene2'] == gene])
        connections[idx] = float(count1 + count2)
    return connections

def train_stgcn(dataset,val_ratio=0.2):
    args = Args_miRNA()
    args.n_vertex = dataset.num_nodes
    n_vertex = dataset.num_nodes

    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} sequences")

    # GSO
    edge_index = sequences[0][0].edge_index
    edge_weight = sequences[0][0].edge_attr.squeeze() if sequences[0][0].edge_attr is not None else None

    adj = torch.zeros((args.n_vertex, args.n_vertex)) # symmetric matrix
    adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight # diagonal vs nondiagonal elements for adj matrix
    D = torch.diag(torch.sum(adj, dim=1) ** (-0.5))
    args.gso = torch.eye(args.n_vertex) - D @ adj @ D
    
    train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx = dataset.split_sequences(sequences, labels)
    
    with open('plottings_STEPmi/split_indices.txt', 'w') as f:
        f.write("Train Indices:\n")
        f.write(", ".join(map(str, train_idx)) + "\n")
        f.write("\nValidation Indices:\n")
        f.write(", ".join(map(str, val_idx)) + "\n")

    gene_connections = compute_gene_connections(dataset)

    model = STEPmi(args, args.blocks, args.n_vertex, gene_connections)
    model = model.float() # convert model to float to solve type error

    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    num_epochs = 60
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    save_dir = 'plottings_STEPmi'
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_stats = []
        all_targets = []
        all_outputs = []

        for seq,label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            x,target = process_batch(seq, label)
            output = model(x)
   
            batch_stats.append({
                'target_range': [target.min().item(), target.max().item()],
                'output_range': [output.min().item(), output.max().item()],
                'target_mean': target.mean().item(),
                'output_mean': output.mean().item()
            })

            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())

            loss = miRNA_enhanced_temporal_loss(
             output[:, :, -1:, :],
             target,
             x
            )

            #loss = criterion(output[:, :, -1:, :], target)
            if torch.isnan(loss):
                print("NaN loss detected!")
                print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        val_loss_total = 0
        all_val_targets = []
        all_val_outputs = []

        with torch.no_grad():
            for seq,label in zip(val_sequences, val_labels):
                x,target = process_batch(seq, label)
                
                output = model(x)
                all_val_targets.append(target.detach().cpu().numpy())
                all_val_outputs.append(output[:, :, -1:, :].detach().cpu().numpy())
            
                val_loss = miRNA_enhanced_temporal_loss(output[:, :, -1:, :], target, x)
                val_loss_total += val_loss.item()

        avg_train_loss = total_loss / len(train_sequences)
        avg_val_loss = val_loss_total / len(val_sequences)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)
        
        # Handle overfitting / dataset is limited
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'{save_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
    checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=True)
    #checkpoint = torch.load(f'{save_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'{save_dir}/training_progress.png')
    plt.close()
    
    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels

def evaluate_model_performance(model, val_sequences, val_labels, dataset,save_dir='plottings_STEPmi'):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)
            output = model(x)
    
            # only the expression predictions
            output = output[:, :, -1:, :].squeeze().cpu().numpy()  # [nodes] expression values
            target = target.squeeze().cpu().numpy()  # [nodes] expression values
            
            all_predictions.append(output)
            all_targets.append(target)

    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)      # [time_points, nodes]
    
    overall_metrics = calculate_overall_metrics(predictions, targets)
    gene_metrics = calculate_gene_metrics(predictions, targets, dataset)

    create_evaluation_plots(predictions, targets, dataset, save_dir)
    
    metrics = {
        'Overall': overall_metrics,
        'Gene': gene_metrics
    }
    
    return metrics

def calculate_overall_metrics(predictions, targets):
    
    metrics = {}

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, dataset):
    """
    Calculate gene-specific metrics for temporal correlation per gene over time
    """
    
    metrics = {}
    genes = list(dataset.node_map.keys())
    
    # Per-gene correlations
    gene_correlations = []
    gene_rmse = []
    gene_spearman_correlations = []

    for gene_idx, gene in enumerate(genes):
        pred_gene = predictions[:, gene_idx]  # All timepoints for this gene
        true_gene = targets[:, gene_idx]
        
        corr, _ = pearsonr(pred_gene, true_gene)
        spearman_corr, spearman_p = spearmanr(pred_gene, true_gene)
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        
        gene_correlations.append((gene, corr))
        gene_spearman_correlations.append((gene, spearman_corr))
        gene_rmse.append(rmse)
    
    # Sort genes by correlation
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    gene_spearman_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations])
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}
    
    return metrics

def create_gene_temporal_plots(predictions, targets, dataset, save_dir):
   
    genes = list(dataset.node_map.keys())
    genes_per_page = 15  # Show 15 genes per page (5x3 grid)
    num_genes = len(genes)
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page
    
    for page in range(num_pages):
        plt.figure(figsize=(20, 15))
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        current_genes = genes[start_idx:end_idx]
        
        for i, gene in enumerate(current_genes):
            plt.subplot(5, 3, i+1)
            gene_idx = dataset.node_map[gene]
            
            # æactual and predicted values
            plt.plot(targets[:, gene_idx], 'b-', label='Actual', marker='o')
            plt.plot(predictions[:, gene_idx], 'r--', label='Predicted', marker='s')
            
            # metrics for specific gene
            corr, _ = pearsonr(targets[:, gene_idx], predictions[:, gene_idx])
            rmse = np.sqrt(mean_squared_error(targets[:, gene_idx], predictions[:, gene_idx]))
            
            # Calculate changes
            actual_changes = np.diff(targets[:, gene_idx])
            pred_changes = np.diff(predictions[:, gene_idx])
            direction_acc = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            
            plt.title(f'Gene: {gene}\nCorr: {corr:.3f}, RMSE: {rmse:.3f}\nDir Acc: {direction_acc:.3f}')
            plt.xlabel('Time Step')
            plt.ylabel('Expression')
            if i == 0:  
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temporal_patterns_page_{page+1}.png')
        plt.close()

def create_evaluation_plots(predictions, targets, dataset, save_dir):

    # 1. Overall prediction scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction Performance')
    plt.savefig(f'{save_dir}/overall_scatter.png')
    plt.close()
    
    # 2. Change distribution plot
    true_changes = np.diff(targets, axis=0).flatten()
    pred_changes = np.diff(predictions, axis=0).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.hist(true_changes, bins=50, alpha=0.5, label='Actual Changes')
    plt.hist(pred_changes, bins=50, alpha=0.5, label='Predicted Changes')
    plt.xlabel('Expression Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Expression Changes')
    plt.legend()
    plt.savefig(f'{save_dir}/change_distribution.png')
    plt.close()
    
    # 3. Gene temporal patterns for all genes
    create_gene_temporal_plots(predictions, targets, dataset, save_dir)

def plot_gene_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset, save_dir='plottings_STEPmi', genes_per_page=12):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels
    
    num_genes = dataset.num_nodes
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(all_sequences, all_labels):
            x, target = process_batch(seq, label)
            output = model(x)
            output = output[:, :, -1:, :].squeeze().cpu().numpy() 
            target = target.squeeze().cpu().numpy()  
            all_predictions.append(output)
            all_targets.append(target)
    
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)          # [time_points, nodes]
    
    gene_names = list(dataset.node_map.keys())
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(20, 15))  
        
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        page_genes = gene_names[start_idx:end_idx]
        
        for i, gene_name in enumerate(page_genes):
            gene_idx = start_idx + i
            rows = (genes_per_page + 1) // 2
            plt.subplot(rows, 2, i + 1) 
        
            train_time_points = range(len(train_labels))
            plt.plot(train_time_points, targets[:len(train_labels), gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_time_points, predictions[:len(train_labels), gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            val_time_points = range(len(train_labels), len(train_labels) + len(val_labels))
            plt.plot(val_time_points, targets[len(train_labels):, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_time_points, predictions[len(train_labels):, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            # Set larger font sizes
            plt.title(f'Gene: {gene_name}', fontsize=16)
            plt.xlabel('Time Points', fontsize=14)
            plt.ylabel('Expression Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.pdf', dpi=900)
        plt.close()


def plot_gene_predictions_train_val_proper_label(
    model, 
    train_sequences, 
    train_labels, 
    val_sequences, 
    val_labels, 
    dataset,
    save_dir='plottings_STEPmi', 
    genes_per_page=12
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels
    
    num_genes = dataset.num_nodes
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(all_sequences, all_labels):
            x, target = process_batch(seq, label)
            output = model(x)
            output = output[:, :, -1:, :].squeeze().cpu().numpy() 
            target = target.squeeze().cpu().numpy()  
            all_predictions.append(output)
            all_targets.append(target)
    
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)          # [time_points, nodes]
    
    gene_names = list(dataset.node_map.keys())
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(20, 15))  
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        page_genes = gene_names[start_idx:end_idx]
        
        for i, gene_name in enumerate(page_genes):
            gene_idx = start_idx + i
            rows = (genes_per_page + 1) // 2  
            plt.subplot(rows, 2, i + 1) 
            
            train_time_points = range(len(train_labels))
            plt.plot(train_time_points, targets[:len(train_labels), gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_time_points, predictions[:len(train_labels), gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            val_time_points = range(len(train_labels), len(train_labels) + len(val_labels))
            plt.plot(val_time_points, targets[len(train_labels):, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_time_points, predictions[len(train_labels):, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            plt.title(f'Gene: {gene_name}')
            plt.xlabel('Time Points')
            plt.ylabel('Expression Value')
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize='small', frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.png')
        plt.close()

def get_predictions_and_targets(model, val_sequences, val_labels):
    """
    Extract predictions and targets from validation data.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)
            output = model(x)
            
            # last time step for predictions
            output = output[:, :, -1:, :]
            
            pred = output.squeeze().cpu().numpy()  # Should be [number of genes] or [1, number of genes]
            true = target.squeeze().cpu().numpy() 
            
            if len(pred.shape) == 1:
                pred = pred.reshape(1, -1)
            if len(true.shape) == 1:
                true = true.reshape(1, -1)
            
            all_predictions.append(pred)
            all_targets.append(true)

    predictions = np.vstack(all_predictions)  # Should be [total test time points, number of genes]
    targets = np.vstack(all_targets)        # Should be [total test time points, 52]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return predictions, targets

def analyze_gene_characteristics(dataset, predictions, targets):
    """
    Analyze relationship between gene properties and prediction performance
    """
    genes = list(dataset.node_map.keys())
    
    # Calculate gene correlations
    gene_correlations = {}
    for gene in genes:
        gene_idx = dataset.node_map[gene]
        pred_gene = predictions[:, gene_idx]  # [time_points]
        true_gene = targets[:, gene_idx]
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_correlations[gene] = corr
    
    # Collect gene properties
    gene_stats = {gene: {
        'degree': len(dataset.base_graph[gene]),
        'expression_range': None,
        'expression_std': None,
        'correlation': gene_correlations[gene]
    } for gene in genes}
    
    # Calculate expression statistics
    for gene in genes:
        all_expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_values = np.concatenate([gene1_expr, gene2_expr])
            all_expressions.extend(expr_values)
        
        all_expressions = np.array(all_expressions)
        gene_stats[gene].update({
            'expression_range': np.ptp(all_expressions),
            'expression_std': np.std(all_expressions)
        })
    
    plt.figure(figsize=(15, 10))
    
    # 1. Correlation vs Degree
    plt.subplot(2, 2, 1)
    degrees = [gene_stats[gene]['degree'] for gene in genes]
    correlations = [gene_stats[gene]['correlation'] for gene in genes]
    plt.scatter(degrees, correlations)
    plt.xlabel('Number of Interactions')
    plt.ylabel('Prediction Correlation')
    plt.title('Gene Connectivity vs Prediction Performance')
    
    # 2. Correlation vs Expression Range
    plt.subplot(2, 2, 2)
    ranges = [gene_stats[gene]['expression_range'] for gene in genes]
    plt.scatter(ranges, correlations)
    plt.xlabel('Expression Range')
    plt.ylabel('Prediction Correlation')
    plt.title('Expression Variability vs Prediction Performance')
    
    plt.subplot(2, 2, 3)
    plt.hist(correlations, bins=20)
    plt.xlabel('Correlation')
    plt.ylabel('Count')
    plt.title('Distribution of Gene Correlations')
    
    plt.tight_layout()
    plt.savefig('plottings_STEPmi/gene_analysis.png')
    plt.close()
    
    print("\nGene Analysis Summary:")
    print("\nTop 5 Most Connected Genes:")
    sorted_by_degree = sorted(gene_stats.items(), key=lambda x: x[1]['degree'], reverse=True)[:5]
    for gene, stats in sorted_by_degree:
        print(f"{gene}: {stats['degree']} connections, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Most Variable Genes:")
    sorted_by_range = sorted(gene_stats.items(), key=lambda x: x[1]['expression_range'], reverse=True)[:5]
    for gene, stats in sorted_by_range:
        print(f"{gene}: range {stats['expression_range']:.4f}, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Best Predicted Genes:")
    sorted_by_corr = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)[:5]
    for gene, stats in sorted_by_corr:
        print(f"{gene}: correlation {stats['correlation']:.4f}, connections: {stats['degree']}")
    
    print(f"\nCorrelation values for all genes:")
    sorted_by_corr_all_genes = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)
    for gene, stats in sorted_by_corr_all_genes:
        print(f"{gene}: correlation {stats['correlation']:.4f}, connections: {stats['degree']}")
    
    return gene_stats

def analyze_temporal_patterns(dataset, predictions, targets):
    time_points = dataset.time_points
    genes = list(dataset.node_map.keys())

    temporal_stats = {
        'prediction_lag': [],  # Time shift between predicted and actual peaks
        'pattern_complexity': [],  # Number of direction changes
        'prediction_accuracy': []  # Accuracy by time point
    }

    time_point_accuracy = []
    for t in range(len(predictions)):
        corr = pearsonr(predictions[t].flatten(), targets[t].flatten())[0]
        time_point_accuracy.append(corr)
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_point_accuracy)
    plt.xlabel('Time Point')
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy Over Time')
    plt.savefig(f'plottings_STEPmi/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats

class Args_miRNA:
    def __init__(self):
        self.Kt=3
        self.Ks=3
        self.n_his=10
        self.n_pred = 1
       
        self.blocks = [
            [256, 256, 256],
            [256, 224, 224],    
            [224, 192, 192],
            [192, 224, 224],   
            [224, 128, 1]
        ]

        self.act_func = 'gelu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.2

if __name__ == "__main__":
    dataset = TemporalGraphDatasetMirna(
        csv_file = '/Users/beyzakaya/Desktop/STEP/dataset/mirna/interaction_mirna.csv',
        embedding_dim=256,
        seq_len=10,
        pred_len=1
    )

    genes_of_interest = ['MYB', 'Zfp800', 'Nr6a1', 'Hnrnpu']

    embeddings = []
    for t in dataset.time_points:
        emb = dataset.node_features[t].numpy()
        idxs = [dataset.node_map[g] for g in genes_of_interest]
        embeddings.append(emb[idxs])
    embeddings = np.stack(embeddings)  # shape: [num_time_points, num_genes, embedding_dim]
    embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_flat)
    embeddings_2d = embeddings_2d.reshape(len(dataset.time_points), len(genes_of_interest), 2)

    plt.figure(figsize=(10, 8)) 

    for i, gene in enumerate(genes_of_interest):
        plt.plot(embeddings_2d[:, i, 0], embeddings_2d[:, i, 1], marker='o', label=gene)
        for t_idx, t in enumerate(dataset.time_points):
            plt.text(
                embeddings_2d[t_idx, i, 0],
                embeddings_2d[t_idx, i, 1],
                f"{t}",
                fontsize=12 
            )

    plt.title("Temporal Trajectories of Node2Vec Embeddings mRNA", fontsize=18)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig("plottings_STEPmi/gene_embedding_trajectories_mrna.pdf", dpi=900)
    plt.show()
   
    model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_stgcn(dataset, val_ratio=0.2)

    metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset)
    plot_gene_predictions_train_val_proper_label(model, train_sequences, train_labels, val_sequences, val_labels, dataset)

    print("\nModel Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Mean Spearman Correlation: {metrics['Gene']['Mean_Spearman_Correlation']:.4f}")
    print(f"Best Performing Genes Pearson: {', '.join(metrics['Gene']['Best_Genes_Pearson'])}")
    print(f"Best Performing Genes Spearman: {', '.join(metrics['Gene']['Best_Genes_Spearman'])}")

    print("\nTemporal Performance:")
    print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
    print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
    print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
    print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

    predictions, targets = get_predictions_and_targets(model, val_sequences, val_labels)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)

