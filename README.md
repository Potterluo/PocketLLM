# PocketLLM

PocketLLM is an experimental project that explores how to train and deploy ultra-small language models on microcontrollers.

The goal is to build a complete pipeline:

LLM data generation → tiny model training → quantization → deployment on ESP32-class devices.

With only ~20M parameters and INT4 quantization, the model can run on devices with as little as 32MB of memory.

This project demonstrates that conversational AI does not always require GPUs or servers — even a microcontroller can run a tiny language model.
