import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class QuantumAttentionLayer(nn.Module):
    """
    Quantum attention mechanism using variational quantum circuits
    """
    def __init__(self, embed_dim, n_qubits=4, n_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Classical linear layers for projecting to quantum dimension
        self.q_proj = nn.Linear(embed_dim, n_qubits)
        self.k_proj = nn.Linear(embed_dim, n_qubits)
        self.v_proj = nn.Linear(embed_dim, n_qubits)
        self.out_proj = nn.Linear(n_qubits, embed_dim)
        
        # Quantum circuit parameters
        self.n_params = n_layers * n_qubits * 3  # 3 parameters per qubit per layer
        self.quantum_weights = nn.Parameter(torch.randn(self.n_params) * 0.1)
        
        # Create quantum node
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface='torch')
    
    def quantum_circuit(self, inputs, weights):
        """
        Variational quantum circuit for attention computation
        """
        # Encode classical data into quantum states
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RX(weights[param_idx], wires=i)
                qml.RY(weights[param_idx + 1], wires=i)
                qml.RZ(weights[param_idx + 2], wires=i)
                param_idx += 3
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # Circular connectivity
        
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def quantum_attention_scores(self, q, k):
        """
        Compute attention scores using quantum circuits
        """
        batch_size, seq_len, _ = q.shape
        attention_scores = torch.zeros(batch_size, seq_len, seq_len)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Combine query and key for quantum processing
                    combined_input = torch.tanh(q[b, i] + k[b, j])  # Simple combination
                    
                    # Process through quantum circuit
                    quantum_output = self.qnode(combined_input, self.quantum_weights)
                    
                    # Convert quantum output to attention score
                    attention_scores[b, i, j] = torch.sum(torch.stack(quantum_output))
        
        return attention_scores
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to quantum dimensions
        q = torch.tanh(self.q_proj(x))  # Bounded for quantum encoding
        k = torch.tanh(self.k_proj(x))
        v = self.v_proj(x)
        
        # Compute quantum attention scores
        attention_scores = self.quantum_attention_scores(q, k)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores / np.sqrt(self.n_qubits), dim=-1)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, v)
        
        # Final projection
        output = self.out_proj(attended_values)
        
        return output, attention_weights

class ClassicalFeedForward(nn.Module):
    """
    Classical feed-forward network with residual connections
    """
    def __init__(self, embed_dim, ff_dim=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class HybridTransformerBlock(nn.Module):
    """
    Single transformer block with quantum attention and classical feed-forward
    """
    def __init__(self, embed_dim, n_qubits=4, n_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        self.quantum_attention = QuantumAttentionLayer(embed_dim, n_qubits, n_layers)
        self.feed_forward = ClassicalFeedForward(embed_dim, ff_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Quantum attention with residual connection
        attended, attention_weights = self.quantum_attention(x)
        x = self.layer_norm1(x + self.dropout(attended))
        
        # Classical feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x, attention_weights

class QuantumTransformer(nn.Module):
    """
    Complete hybrid quantum-classical transformer for text classification
    """
    def __init__(self, vocab_size, embed_dim=128, n_blocks=2, n_qubits=4, 
                 n_classes=4, max_seq_len=50, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Input embeddings (simplified - using TF-IDF features directly)
        self.input_projection = nn.Linear(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(embed_dim, n_qubits, 2, embed_dim*2, dropout)
            for _ in range(n_blocks)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len = x.shape[0], min(x.shape[1], self.max_seq_len)
        
        # Project input features to embedding dimension
        x = x[:, :seq_len]  # Truncate if necessary
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights_list = []
        for block in self.blocks:
            x, attention_weights = block(x)
            attention_weights_list.append(attention_weights)
        
        # Global average pooling for classification
        x = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, attention_weights_list

def prepare_data():
    """
    Prepare 20newsgroups dataset for text classification
    """
    print("Loading 20newsgroups dataset...")
    
    # Load a subset of categories for faster training
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                        remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                       remove=('headers', 'footers', 'quotes'))
    
    # Convert to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                               max_df=0.8, min_df=5)
    
    X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
    X_test = vectorizer.transform(newsgroups_test.data).toarray()
    
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(categories)}")
    
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """
    Train the hybrid quantum-classical transformer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Reshape data for sequence processing
            data = data.unsqueeze(1).expand(-1, 10, -1)  # Create sequence dimension
            
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            
            # Gradient clipping for quantum parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.unsqueeze(1).expand(-1, 10, -1)
                
                logits, _ = model(data)
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_accuracies

def visualize_attention(model, sample_data, tokenizer_features):
    """
    Visualize quantum attention patterns
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        sample_data = sample_data.to(device)
        sample_data = sample_data.unsqueeze(1).expand(-1, 10, -1)
        
        logits, attention_weights_list = model(sample_data)
        
        # Visualize attention from the first layer
        attention = attention_weights_list[0][0].cpu().numpy()  # First sample, first layer
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Attention heatmap
        plt.subplot(2, 2, 1)
        plt.imshow(attention, cmap='viridis', aspect='auto')
        plt.title('Quantum Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.colorbar()
        
        # Plot 2: Attention patterns across heads (averaged)
        plt.subplot(2, 2, 2)
        avg_attention = np.mean(attention, axis=0)
        plt.plot(avg_attention, 'o-')
        plt.title('Average Attention per Position')
        plt.xlabel('Position')
        plt.ylabel('Attention Weight')
        
        # Plot 3: Quantum parameter distribution
        plt.subplot(2, 2, 3)
        quantum_params = []
        for block in model.blocks:
            quantum_params.extend(block.quantum_attention.quantum_weights.detach().cpu().numpy())
        plt.hist(quantum_params, bins=20, alpha=0.7)
        plt.title('Quantum Parameter Distribution')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        
        # Plot 4: Classification probabilities
        plt.subplot(2, 2, 4)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        plt.bar(categories, probs)
        plt.title('Classification Probabilities')
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution function
    """
    print("=== Hybrid Quantum-Classical Transformer ===\n")
    
    # Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data()
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = QuantumTransformer(
        vocab_size=X_train.shape[1],
        embed_dim=64,
        n_blocks=2,
        n_qubits=4,
        n_classes=4,
        max_seq_len=10,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Quantum parameters: {sum(p.numel() for name, p in model.named_parameters() if 'quantum_weights' in name)}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, epochs=5, lr=0.001)
    
    # Final evaluation
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(1).expand(-1, 10, -1)
            
            logits, _ = model(data)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    final_accuracy = 100 * correct / total
    print(f'\nFinal Test Accuracy: {final_accuracy:.2f}%')
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Visualize attention patterns
    sample_data = X_test_tensor[:1]  # First test sample
    visualize_attention(model, sample_data, vectorizer)
    
    print("\n=== Analysis Complete ===")
    print(f"The hybrid quantum-classical transformer achieved {final_accuracy:.2f}% accuracy")
    print("Key features:")
    print("- Quantum attention mechanism using variational circuits")
    print("- Classical feed-forward layers for processing")
    print("- Hybrid architecture combining quantum and classical components")
    print("- Demonstrable attention patterns and quantum parameter evolution")

if __name__ == "__main__":
    main()
