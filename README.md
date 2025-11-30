<h1 align="center">GlassBoxViewer</h1>

<p align="center">
  Real-time Visualizer for Neural Networks
</p>


## About
This project aims to bring visualization to neural networks, to see through the black box that is machine learning, literally. This is a slow burn project that I will continue to work on as I have time to do so. This is a more of a demo/proof of design approach towards creating a visualizer for neural networks. Currently, this for certain works for LLM GPT2 and CNN Resnet50.

## How To Run
```
pip install vispy
pip install PyQt6
pip install transformers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
pipreqs was giving me trouble so please pip install the above. This project should be able to run on the CPU just as fine.
Then run one of the two scripts with the leading name "example_" and the visualizer will take a few moments like in the demo below to appear. Closing just the visualizer window will not stop the program so please stop the script itself, this will be a future update for sure.



## Demo Usage

https://github.com/user-attachments/assets/36b4c2c7-f1b1-485d-8201-7800a6956dfb


## LLM GPT2 Example Demo
<h3 align="center">Linear Method</h3>

https://github.com/user-attachments/assets/4753955b-9cbd-47de-ab5a-313034fbe5d5

<h3 align="center">Ring Method</h3>

https://github.com/user-attachments/assets/4a788a41-bc27-4122-8f24-50778398bb94



## CNN ResNet50 Example Demo
<h3 align="center">Linear Method</h3>

https://github.com/user-attachments/assets/cc083425-ce5e-4026-acce-10bca9ceef7d

<h3 align="center">Ring Method</h3>


https://github.com/user-attachments/assets/735a6034-9351-48ab-9a73-becd943652f0




## Future Plans
Ultimately, I plan to make a visualizer to run with inference engines so anyone can have a nice visual display of how the neural network is working on the inside.
- Make this usable more than one batch, should be doable to add, most likely adding another color for every batch to run in tandem in the same visualizer

  
## Known Issues
- Warning about the data transfer in the queue. Needs a better way to deal with the unsafe pickle I believe that does not slow down the process
- Program does not stop if the visualizer is closed
- Current example_ scripts are barebones to show the visualizer, later example_ scripts will be more useful to use and modified quickly 

