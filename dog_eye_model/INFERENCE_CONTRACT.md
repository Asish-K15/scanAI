\# ScanAI Dog Eye Inference Contract



\## Model



EfficientNet-B0



Model version: dog-eye-v1



Inference engine: ONNX Runtime



\## Input



RGB image



Image size: 224 x 224



Normalization:



Mean:

\[0.485, 0.456, 0.406]



Standard deviation:

\[0.229, 0.224, 0.225]



\## Supported Conditions



The model currently supports:



1\. conjunctivitis

2\. entropion



\## Output



The inference module returns:



```json

{

&#x20; "condition": "entropion",

&#x20; "confidence": 0.99966,

&#x20; "confidence\_level": "high",

&#x20; "uncertain": false,

&#x20; "probabilities": {

&#x20;   "conjunctivitis": 0.00034,

&#x20;   "entropion": 0.99966

&#x20; },

&#x20; "model": "EfficientNet-B0",

&#x20; "model\_version": "dog-eye-v1",

&#x20; "engine": "ONNX Runtime",

&#x20; "screening\_only": true

}

