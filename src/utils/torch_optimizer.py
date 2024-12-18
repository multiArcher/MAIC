from types import ModuleType

import torch
import numpy


def optimize_tensor_display(
        torch_module: ModuleType,
        custom_repr: bool = True,
        add_display_array: bool = True,
    ) -> None:
    """Custom tensor display function

    Args:
        torch_module: Torch module to be modified.
        custom_repr: Whether to add dimension to repr.
        add_display_array: Whether to add numpy_array attribute for dataview displaying in Pycharm.
    """
    if custom_repr:
        _custom_repr(torch_module.Tensor)
    if add_display_array:
        _add_display_array(torch_module.Tensor)

def _custom_repr(tensor_class: torch._C._TensorMeta) -> None:
    """Use Customized Tensor repr.

    Args:
        tensor_class (torch._C._TensorMeta): torch.Tensor class to be modified.
    """
    assert type(tensor_class) == torch._C._TensorMeta   # Make sure a correct modifing to torch.Tensor.

    tensor_class.old_repr = tensor_class.__repr__

    def _shape_repr(self: torch.Tensor) -> str:
        # Add Tensor shape to repr
        new_text = [f"torch.Tensor(shape={tuple(self.shape)}"]  # Show shape
        
        if self.is_cuda:
            # Show device
            new_text.append(f", device={self.device.type}:{self.device.index}")
        
        new_text.append(f", dtype={str(self.dtype)}") # Show dtype.

        new_text.append(")\n")
        new_text.append(self.old_repr())    # Add original text.
            
        return "".join(new_text)
    
    tensor_class.__repr__ = _shape_repr


def _add_display_array(tensor_class: torch._C._TensorMeta) -> None:
    """
    Add a ndarray with tensor data for displaying in Pycharm DataViewer which
    does not support cuda tensors.

    Args:
        tensor_class (torch._C._TensorMeta): torch.Tensor class to be modified.
    """
    assert type(tensor_class) == torch._C._TensorMeta   # Make sure a correct modifing to torch.Tensor.

    def display_array(self: torch.Tensor) -> numpy.ndarray:
        # Return a ndarray with tensordata.
        return self.cpu().detach().numpy()

    tensor_class.display_array = property(display_array)

    
if __name__ == '__main__':
    optimize_tensor_display(torch)

    a = torch.tensor([1, 2, 3], device=torch.device("cuda"))
    
    print(a)
    