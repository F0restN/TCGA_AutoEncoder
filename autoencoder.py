import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

torch.manual_seed(42)
np.random.seed(42)

# Define autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.2):
        """
        Initialize autoencoder model
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout_rate: Dropout rate for regularization
        """
        super(Autoencoder, self).__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final encoding layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse traverse hidden dimensions
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final decoding layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input data to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space back to original space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode then decode"""
        z = self.encode(x)
        return self.decode(z)

# Data loading and preprocessing function
def load_and_preprocess_data(file_path):
    """
    Load and preprocess TCGA data
    
    Args:
        file_path: Path to TCGA data file
    
    Returns:
        Processed data and sample information
    """
    # Read data
    data = pd.read_csv(file_path, sep='\t')
    
    # Extract sample IDs and gene expression data
    sample_ids = data.columns[1:]
    gene_ids = data.iloc[:, 0]
    expression_data = data.iloc[:, 1:].T  # Transpose to make samples as rows, genes as columns
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expression_data)
    
    return scaled_data, sample_ids, gene_ids

# Function to train autoencoder
def train_autoencoder(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    """
    Train autoencoder model
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Training and validation loss history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Visualize training process
def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()

# Dimensionality reduction and visualization function
def visualize_latent_space(model, data_loader, sample_ids=None):
    """
    Visualize latent space
    
    Args:
        model: Trained autoencoder model
        data_loader: Data loader
        sample_ids: List of sample IDs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get latent space representations
    latent_vectors = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            latent = model.encode(inputs)
            latent_vectors.append(latent.cpu().numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 10))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.7)
    
    if sample_ids is not None and len(sample_ids) == latent_tsne.shape[0]:
        # Extract sample type information (assuming sample IDs contain type info, e.g., TCGA-XX-XXXX-01A for tumor)
        sample_types = []
        for sample_id in sample_ids:
            if '-01' in str(sample_id):  # Tumor sample
                sample_types.append('Tumor')
            elif '-11' in str(sample_id):  # Normal sample
                sample_types.append('Normal')
            else:
                sample_types.append('Other')
        
        # Color by sample type
        plt.figure(figsize=(12, 10))
        for sample_type in set(sample_types):
            indices = [i for i, x in enumerate(sample_types) if x == sample_type]
            plt.scatter(latent_tsne[indices, 0], latent_tsne[indices, 1], label=sample_type, alpha=0.7)
        
        plt.legend()
    
    plt.title('t-SNE Visualization of Latent Space')
    plt.savefig('latent_space_tsne.png')
    plt.show()
    
    # If latent space dimension > 2, also visualize using PCA
    if latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.7)
        plt.title('PCA Visualization of Latent Space')
        plt.savefig('latent_space_pca.png')
        plt.show()

# Feature importance analysis
def analyze_feature_importance(model, gene_ids):
    """
    Analyze gene feature importance
    
    Args:
        model: Trained autoencoder model
        gene_ids: List of gene IDs
    """
    # Get weights from first encoder layer
    first_layer_weights = model.encoder[0].weight.data.cpu().numpy()
    
    # Calculate importance as sum of absolute weights for each input feature
    feature_importance = np.sum(np.abs(first_layer_weights), axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Gene': gene_ids,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Visualize top 20 most important genes
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Gene', data=importance_df.head(20))
    plt.title('Top 20 Most Important Genes')
    plt.tight_layout()
    plt.savefig('gene_importance.png')
    plt.show()
    
    return importance_df

# Main function
def main():
    # Data file path
    file_path = 'TCGA.COADREAD.sampleMap_HiSeqV2_exon'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data, sample_ids, gene_ids = load_and_preprocess_data(file_path)
    
    # Split training and test sets
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create PyTorch datasets and data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define model parameters
    input_dim = data.shape[1]  # Number of genes
    hidden_dims = [1024, 512, 256]  # Hidden layer dimensions
    latent_dim = 64  # Latent space dimension
    
    # Create autoencoder model
    print(f"Creating autoencoder model with input dimension: {input_dim}")
    model = Autoencoder(input_dim, hidden_dims, latent_dim)
    
    # Train model
    print("Starting model training...")
    train_losses, val_losses = train_autoencoder(
        model, train_loader, test_loader, num_epochs=100, learning_rate=1e-3
    )
    
    # Visualize training process
    plot_training_history(train_losses, val_losses)
    
    # Visualize latent space
    print("Visualizing latent space...")
    full_dataset = TensorDataset(torch.FloatTensor(data))
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    visualize_latent_space(model, full_loader, sample_ids)
    
    # Analyze feature importance
    print("Analyzing gene importance...")
    importance_df = analyze_feature_importance(model, gene_ids)
    
    # Save model
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Model saved as 'autoencoder_model.pth'")
    
    # Save important genes list
    importance_df.to_csv('gene_importance.csv', index=False)
    print("Gene importance saved as 'gene_importance.csv'")

if __name__ == "__main__":
    main()
