from pydantic import BaseModel, Field, validator


# Base class with validator
class BaseModelConfig(BaseModel):
    model_type: str

    @validator("model_type")
    def validate_model_type(cls, value):
        if value not in ["scale-mae", "croma", "dofa"]:
            raise ValueError(f"Unsupported model type: {value}")
        return value

    class Config:
        validate_assignment = True

# Subclass inheriting from BaseModelConfig
class ExtendedModelConfig(BaseModelConfig):
    model_type: str = 'dofa2'
    additional_feature: str = "default"

# Test instantiation
extended_config = ExtendedModelConfig()  # Validator runs and passes
extended_config.model_type="dofa3"
#extended_config.model_type = "dofa2"  # Validator runs and passes

