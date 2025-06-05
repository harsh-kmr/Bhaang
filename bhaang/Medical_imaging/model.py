import torch
from transformers import AutoModel
from pprint import pprint
import torchvision
#from torchvision.models import ResNet18_Weights
from tabulate import tabulate


class Model_master():
    def __init__(self, model_provider):
        self.model_provider = model_provider
    
    def get_model_torch(self, repo_id, model_name, *args, **kwargs):
        try:
            self.model = torch.hub.load( repo_id, model_name, *args, **kwargs)
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from Torch Hub: {e}")

    def get_model_huggingface(self, model_name, *args, **kwargs):
        try:
            self.model = AutoModel.from_pretrained(model_name, *args, **kwargs)
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from Hugging Face Hub: {e}")
    
    def get_model_local(self, model_path, *args, **kwargs):
        """
        Loads the full model saved using torch.save.
        """
        try:
            self.model = torch.load(model_path, *args, **kwargs, weights_only=False)
            self.model.eval()
            print("Full model loaded successfully.")
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load full model: {e}")


    
    def get_model(self, repo_id, model_name, *args, **kwargs):
        """
        Load model from the specified provider
        Args:
            repo_id (str): Repository ID for Torch Hub or local path of the saved model
            model_name (str): Model name for Torch Hub or Hugging Face Hub
            *args: Additional arguments to pass to the model loading function
            **kwargs: Additional keyword arguments to pass to the model loading function
        Returns:
            model: Loaded model"""
        if self.model_provider == 'torch':
            return self.get_model_torch(repo_id, model_name, *args, **kwargs)
        elif self.model_provider == 'huggingface':
            return self.get_model_huggingface(model_name, *args, **kwargs)
        elif self.model_provider == 'local':
            return self.get_model_local(repo_id, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
        
    def get_accurate_param_count(self):
        """
        Get accurate parameter count without double-counting nested modules
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return total_params, trainable_params
        
    def display_model_layers(self, sample_input=None):
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        print(f"\nModel Summary: {self.model.__class__.__name__}")
        print("-" * 80)
        # move the model to CPU for summary
        self.model.cpu()
        sample_input = sample_input.cpu() if sample_input is not None else None
        
        total_params , trainable_params = self.get_accurate_param_count()
        table_data = []
        hooks = []
        layer_shapes = {}

        def hook_fn(module, inp, outp, name):
            try:
                input_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
                output_shape = outp.shape if hasattr(outp, 'shape') else "Unknown"
            except:
                input_shape, output_shape = "Unknown", "Unknown"
            layer_shapes[name] = (input_shape, output_shape)

        if sample_input is not None:
            for name, layer in self.model.named_modules():
                if isinstance(layer, torch.nn.Module) and name:
                    hook = layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                    hooks.append(hook)
            try:
                _ = self.model(sample_input)
            except Exception as e:
                print(f"Error running forward pass with sample input: {e}")
                for hook in hooks:
                    hook.remove()
                return
            for hook in hooks:
                hook.remove()
        
        for name, layer in self.model.named_modules():
            if name:  # Avoid printing empty layer names
                num_params = sum(p.numel() for p in layer.parameters(recurse=False) )
                input_shape, output_shape = layer_shapes.get(name, ("Unknown", "Unknown"))
                table_data.append([name, layer.__class__.__name__, num_params, input_shape, output_shape])
        
        print(tabulate(table_data, headers=["Layer Name", "Layer Type", "Parameters", "Input Shape", "Output Shape"], tablefmt="grid"))
        print("-" * 80)
        print(f"Total Trainable Parameters: {trainable_params:,}")
        print("-" * 80)
        print("total params: ", total_params)
    

    def add_classification_head(self, num_classes, layer_name = None, in_features=None):
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        #print(layer_name)

        
        if layer_name is None:
            # Find the last fully connected (linear) layer
            last_layer_name, last_layer = None, None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    last_layer_name, last_layer = name, module
                    #print(f"Found fully connected layer '{last_layer_name}' with {last_layer.in_features} input features.")
                    break
            if last_layer_name is None:
                raise ValueError("No fully connected layer found in the model.")
        else:
            if not hasattr(self.model, layer_name):
                raise ValueError(f"Layer '{layer_name}' not found in model.")
            last_layer_name, last_layer = layer_name, getattr(self.model, layer_name)
        
        if in_features is None:
            if not hasattr(last_layer, 'in_features'):
                raise ValueError("Could not determine input features for the new classification head.")
            in_features = last_layer.in_features
        
        new_fc = torch.nn.Linear(in_features, num_classes)
        setattr(self.model, last_layer_name, new_fc)
        
        print(f"Replaced layer '{last_layer_name}' with new classification head ({num_classes} classes).")
        return self.model
    
    def save_model(self, file_path):
        """
        Saves the full model (architecture and weights) using torch.save.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        try:
            torch.save(self.model, file_path)
            print(f"Full model saved successfully at {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save full model: {e}")
    
    

        
            
        
# # test the model.py
# model_provider = 'torch'
# model_master = Model_master(model_provider)
# model = model_master.get_model('pytorch/vision', 'resnet18', weights=ResNet18_Weights.DEFAULT)
# sample_input = torch.randn(1, 3, 224, 224)
# model_master.display_model_layers(sample_input)

# # clear the saved model
# del model