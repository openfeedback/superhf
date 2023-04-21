from transformers import PretrainedConfig, AutoConfig
from copy import deepcopy

class RewardModelConfig(PretrainedConfig):
    model_type = 'reward_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_model_config = kwargs.pop('base_model_config', {})
        base_model_type = base_model_config.pop('model_type', None)

        if base_model_config and base_model_type is not None:
            self.base_model_config = AutoConfig.for_model(base_model_type, **base_model_config)

        # Copy the following attributes over for the `Trainer` class to work
        self.hidden_size = self.base_model_config.hidden_size

    @classmethod
    def from_pretrained_base_model(cls, pretrained_model_name_or_path, **kwargs):
        base_model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        kwargs['base_model_config'] = base_model_config.to_dict()
        return cls(**kwargs)
    
    def to_dict(self):
        output = deepcopy(self.__dict__)
        if hasattr(self, 'base_model_config'):
            if isinstance(self.base_model_config, dict):
                output['base_model_config'] = self.base_model_config
            else:
                output['base_model_config'] = self.base_model_config.to_dict() 
        # output['base_model_config'] = self.base_model_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output
    


    # config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    # if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
    #     logger.warning(
    #         f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
    #         f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
    #     )

    # return cls.from_dict(config_dict, **kwargs)