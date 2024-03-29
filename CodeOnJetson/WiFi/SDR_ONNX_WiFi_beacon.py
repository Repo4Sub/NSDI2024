import numpy as np
import scipy.io as scio
import time
import onnxruntime
import adi

def GenerateSymbolMat(symbol_batch):
    _batch_size = symbol_batch.shape[0]
    _signal_dimension = symbol_batch.shape[1]
    _symbol_length = symbol_batch.shape[2]

    symbol_batch_real = np.real(symbol_batch)
    symbol_batch_imag = np.imag(symbol_batch)
    symbol_batch_mat = np.concatenate((symbol_batch_real,symbol_batch_imag), axis = 1).astype('float32')
    
    return symbol_batch_mat, _batch_size, _signal_dimension


# ZigBee
SymbolFilePath = './BeaconPacket.mat'
SymbolMat = 'BeaconPacket'

Symbol_file = scio.loadmat(SymbolFilePath)
Symbol_batch = Symbol_file[SymbolMat]
Symbol_batch = np.expand_dims(Symbol_batch,axis=0)


Beacon_Symbols, batch_size, signal_dimension = GenerateSymbolMat(Symbol_batch)

# Execution Provider of ONNX backend
EP_list = ['CPUExecutionProvider', 'CUDAExecutionProvider']
EPIdx = 1
print("Execution provider is", EP_list[EPIdx])
WiFi_session = onnxruntime.InferenceSession("WiFiMod.onnx",providers=[EP_list[EPIdx]])

ONNX_Input = {WiFi_session.get_inputs()[0].name: Beacon_Symbols}
ONNX_Output = WiFi_session.run(None, ONNX_Input)
ONNX_Output_array = ONNX_Output[0]

# Post processing for complex signals
realSig = ONNX_Output_array[:,:,0]
imagSig = ONNX_Output_array[:,:,1]
cplxSig = realSig + 1j*imagSig
print(cplxSig.shape)
txWaveform = 6*cplxSig[0,:] * 2**14 # Range mapping
print(txWaveform.shape)

# Connect to SDR
sdr = adi.Pluto('ip:192.168.2.1') # or whatever your Pluto's IP is
centralFrequency = 2.462e9
samplingRate = 20e6
sdr.sample_rate = int(samplingRate) # sampling rate
sdr.tx_rf_bandwidth = int(samplingRate) # bandwidth
sdr.tx_lo = int(centralFrequency)
sdr.tx_hardwaregain_chan0 = 0 # Increase to increase tx power, valid range is -90 to 0 dB

# Transmit signals
numFrames = 100
time_delay = 0.5
sdr.tx_destroy_buffer()
sdr.tx_cyclic_buffer = False
for frameIdx in range(numFrames):
    sdr.tx(txWaveform)
    print("Send Beacon!")
    time.sleep(time_delay)