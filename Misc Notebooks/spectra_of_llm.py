import torch
from transformers import AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from datetime import datetime

def load_model(model_name):
    """Load model from Hugging Face."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir='cache')
    return model

def get_mlp_layers(model):
    """Extract MLP layers from the model."""
    mlp_layers = []
    for name, module in model.named_modules():
        if 'mlp' in name.lower() and isinstance(module, torch.nn.Linear):
            mlp_layers.append((name, module))
    return mlp_layers

def compute_spectrum(weight_matrix):
    """Compute singular value spectrum of weight matrix."""
    if isinstance(weight_matrix, torch.Tensor):
        weight_matrix = weight_matrix.detach().cpu().numpy().astype(np.float32)
    s = linalg.svd(weight_matrix, compute_uv=False)
    return s

def create_spectrum_plot(spectra, layer_names, model_name):
    """Create and save a simple plot of weight spectra."""
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (spectrum, name) in enumerate(zip(spectra, layer_names)):
        color = colors[i % len(colors)]
        plt.plot(np.arange(1, len(spectrum) + 1), spectrum, 
                label=name, color=color)
    
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(model_name)
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f'spectrum_{model_name.replace("/", "_")}_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_and_plot_spectra(model_name, num_layers=5):
    """Analyze weight spectra and create plot."""
    try:
        model = load_model(model_name)
        mlp_layers = get_mlp_layers(model)
        print(f"Found {len(mlp_layers)} MLP layers")
        
        if num_layers and num_layers < len(mlp_layers):
            layer_indices = np.linspace(0, len(mlp_layers)-1, num_layers, dtype=int)
            mlp_layers = [mlp_layers[i] for i in layer_indices]
        
        spectra = []
        layer_names = []
        
        for name, layer in mlp_layers:
            print(f"Analyzing layer: {name}")
            spectrum = compute_spectrum(layer.weight)
            spectra.append(spectrum)
            layer_names.append(name)
        
        save_path = create_spectrum_plot(spectra, layer_names, model_name)
        print(f"\nPlot saved to: {save_path}")
        
        return spectra, layer_names
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None, None

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
    spectra, names = analyze_and_plot_spectra(model_name)