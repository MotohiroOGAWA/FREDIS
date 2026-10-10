import torch
import torch.nn as nn
import inspect

class ModelBase(nn.Module):
    """
    Base class for PyTorch models that automatically records constructor arguments.

    ---------------------------------------------------------------
    **How to inherit correctly**
    ---------------------------------------------------------------
    When you define a subclass, make sure to call super().__init__ like this:

        super(SubClass, self).__init__(
            ignore_config_keys=[],
            **{k: v for k, v in locals().items() if k != 'self'}
        )

    This ensures all __init__ arguments are automatically registered
    as instance attributes for config management and reproducibility.
    ---------------------------------------------------------------
    """
    def __init__(self, ignore_config_keys=None, **kwargs):
        super(ModelBase, self).__init__()
        self._ignore_config_keys = ['self', '_ignore_config_keys', '__class__']

        if ignore_config_keys is not None:
            self._ignore_config_keys.extend(ignore_config_keys)

        for key, value in kwargs.items():
            if key in self._ignore_config_keys:
                continue

            # --- check if attribute is a property in subclass ---
            attr = inspect.getattr_static(self.__class__, key, None)

            if isinstance(attr, property):
                # Skip setting attribute if it is defined as @property
                continue

            setattr(self, key, value)


    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by the subclass.")
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_params(self):
        """
        Get model configuration automatically.

        Returns:
            dict: Model configuration.
        """
        # Get all constructor parameter names dynamically
        constructor_params = inspect.signature(self.__init__).parameters
        config_keys = [param for param in constructor_params if param not in self._ignore_config_keys]

        # Extract only the required parameters from instance attributes
        config = {key: getattr(self, key) for key in config_keys if hasattr(self, key)}
        
        return config
    
    @classmethod
    def from_params(cls, config_param):
        """
        Create model from configuration parameters.

        Args:
            config_param (dict): Model configuration parameters.

        Returns:
            BaseModel: Model instance.
        """
        if config_param is None:
            config_param = {}
        return cls(**config_param)