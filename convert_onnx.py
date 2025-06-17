import torch
import models

model_ae = models.load_ae("model_ae_16_interp.torch")
model_lstm = models.load_lstm_onnx("model_lstm_16_interp.torch")
model_ae.eval()
model_lstm.eval()

example_encoder_inputs = (torch.randn(1, 3, 96, 96),)
example_decoder_inputs = (torch.randn(1, 16),)
example_lstm_inputs = (
    torch.randn(1, 1, 20), 
    torch.zeros(3, 1, 128),
    torch.zeros(3, 1, 128)
)

print(f"Example encoder input shape: {example_encoder_inputs[0].shape}")
example_encoder_outputs = model_ae.encoder(*example_encoder_inputs)
print(f"Example encoder output shape: {example_encoder_outputs.shape}")

print(f"Example decoder input shape: {example_decoder_inputs[0].shape}")
example_decoder_outputs = model_ae.decoder(*example_decoder_inputs)
print(f"Example decoder output shape: {example_decoder_outputs.shape}")

print(f"Example LSTM input shape: {example_lstm_inputs[0].shape}")
example_lstm_outputs = model_lstm(*example_lstm_inputs)
print(f"Example LSTM output shape: {example_lstm_outputs[0].shape}")

enc_params = sum(p.numel() for p in model_ae.encoder.parameters())
dec_params = sum(p.numel() for p in model_ae.decoder.parameters())
lstm_params = sum(p.numel() for p in model_lstm.parameters())

print(f"AE Encoder parameters: {enc_params/1e6}M")
print(f"AE Decoder parameters: {dec_params/1e6}M")
print(f"LSTM parameters: {lstm_params/1e6}M")
print(f"Total parameters: {(enc_params+dec_params+lstm_params)/1e6}M")

encoder_onnx = torch.onnx.export(
    model_ae.encoder, 
    example_encoder_inputs, 
    "onnx_encoder.onnx",
    input_names=['input'],
    output_names=['output']
)
decoder_onnx = torch.onnx.export(
    model_ae.decoder, 
    example_decoder_inputs, 
    "onnx_decoder.onnx",
    input_names=['input'],
    output_names=['output']
)
lstm_onnx = torch.onnx.export(
    model_lstm, 
    example_lstm_inputs, 
    "onnx_lstm.onnx",
    input_names=['input', 'h0', 'c0'],
    output_names=['output', 'hn', 'cn']
)