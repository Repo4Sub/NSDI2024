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
SymbolFilePath = './QPSK_Symbols.mat'
SymbolMat = 'QPSK_Symbols'

Symbol_file = scio.loadmat(SymbolFilePath)
Symbol_batch = Symbol_file[SymbolMat]
Symbol_batch = np.expand_dims(Symbol_batch,axis=0)


QPSK_Symbols, batch_size, signal_dimension = GenerateSymbolMat(Symbol_batch)

# Execution Provider of ONNX backend
EP_list = ['CPUExecutionProvider', 'CUDAExecutionProvider']
EPIdx = 1
print("Execution provider is", EP_list[EPIdx])
QPSK_session = onnxruntime.InferenceSession("QPSK_Modulator.onnx",providers=[EP_list[EPIdx]])

ONNX_Input = {QPSK_session.get_inputs()[0].name: QPSK_Symbols}
ONNX_Output = QPSK_session.run(None, ONNX_Input)
ONNX_Output_array = ONNX_Output[0]

# Post processing for Offset-QPSK
zeroPadding = np.zeros((batch_size,2))
realSig = np.concatenate([ONNX_Output_array[:,:,0], zeroPadding], axis=1)
imagSig = np.concatenate([zeroPadding, ONNX_Output_array[:,:,1]], axis=1)
cplxSig = realSig + 1j*imagSig

# Configure Pluto SDR
sdr = adi.Pluto('ip:192.168.2.1') # or whatever your Pluto's IP is
centralFrequency = 2.405e9
samplingRate = 4e6
sdr.sample_rate = int(samplingRate) # sampling rate
sdr.tx_rf_bandwidth = int(samplingRate) # bandwidth
sdr.tx_lo = int(centralFrequency)
sdr.tx_hardwaregain_chan0 = 0 # Increase to increase tx power, valid range is -90 to 0 dB
txWaveform = cplxSig * 2**14 # Range mapping

# Transmit signals
numFrames = 100
for pktIdx in range(batch_size):
    samples = txWaveform[pktIdx,:]
    for frameIdx in range(numFrames):
        sdr.tx(samples)
        time.sleep(0.5)
        print("Send packet No.", frameIdx)
