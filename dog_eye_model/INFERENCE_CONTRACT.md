# ScanAI Dog Eye Inference Contract



## Model



EfficientNet-B0



Model version: dog-eye-v1



Inference engine: ONNX Runtime



## Input



RGB image



Image size: 224 x 224



Normalization:



Mean:

[0.485, 0.456, 0.406]



Standard deviation:

[0.229, 0.224, 0.225]



## Supported Conditions



The model currently supports:



1\. conjunctivitis

2\. entropion



## Output



The inference module returns:



```json

{

  "condition": "entropion",

  "confidence": 0.99966,

  "confidence_level": "high",

  "uncertain": false,

  "probabilities": {

    "conjunctivitis": 0.00034,

    "entropion": 0.99966

  },

  "model": "EfficientNet-B0",

  "model_version": "dog-eye-v1",

  "engine": "ONNX Runtime",

  "screening_only": true

}

