import torch


def to_tensor(device="cpu"):
    def inner(a, *args):
        # Create a new list with properly converted tensors
        result = []
        for item in a:
            if torch.is_tensor(item):
                result.append(item.to(device=device))
            else:
                result.append(torch.tensor(item).to(device=device))
        return result

    return inner


class to_tensor_clean:
    def __init__(self, device="cpu", debug=False):
        self.device = device
        self.debug = debug

    def __call__(self, a):
        # Extract parameters once to avoid repeated dictionary lookups
        params = a["params"]
        
        # Extract angle values directly
        angle = params["angle"]
        params["ra"] = angle[0]
        params["dec"] = angle[1]
        params["psi"] = angle[2]
        
        # Use pop to remove items (more efficient than del)
        params.pop("angle")
        params.pop("sam_p")
        
        if self.debug:  # Simplified condition
            print(params)
            
        params.move_to_end("spec")
        
        # Convert to tensors more efficiently
        signal_tensor = torch.tensor(a["signal"], device=self.device)
        
        # Use isinstance for type checking (more idiomatic)
        param_values = [v for v in params.values() if not isinstance(v, dict)]
        param_tensor = torch.tensor(param_values, device=self.device)
        
        return signal_tensor, param_tensor
