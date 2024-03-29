import numpy as np
import scipy.io as scio
import time
import onnxruntime
EP_list = ['CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider']
EPIdx = 1
print("Execution provider is", EP_list[EPIdx])
QAM_RRC_session = onnxruntime.InferenceSession("QAMConvMod.onnx",providers=[EP_list[EPIdx]])

# QAM_RRC_session = onnxruntime.InferenceSession("QAMConvMod.onnx",providers=['CUDAExecutionProvider'])
# QAM_RRC_session = onnxruntime.InferenceSession("QAMConvMod.onnx",providers=['TensorrtExecutionProvider'])

def GenerateSymbolMat(MatFile,MatName):
    Symbol_file = scio.loadmat(MatFile)
    Symbol_batch = Symbol_file[MatName]
    _batch_size = Symbol_batch.shape[0]
    _signal_dimension = Symbol_batch.shape[1]
    _symbol_length = Symbol_batch.shape[2]
    _complex_symbol = np.iscomplexobj(Symbol_batch)
    if _complex_symbol:
        print("Input has complex symbols!")
        Symbol_batch_real = np.real(Symbol_batch)
        Symbol_batch_imag = np.imag(Symbol_batch)
        Symbol_batch_mat = np.concatenate((Symbol_batch_real,Symbol_batch_imag), axis = 1).astype('float32')
    else:
        print("Input has real symbols!")
        Symbol_batch_mat = Symbol_batch.astype('float32')
    # Symbol_batch_tensor = torch.tensor(Symbol_batch_mat)
    return Symbol_batch_mat, _complex_symbol, _batch_size, _signal_dimension, _symbol_length

# QAM
SymbolFilePath = './InputSymbol/QAMSymbol_batch_test.mat'
SymbolMat = 'QAMSymbol_batch_test'
QAM_symbol_mat, complex_symbol, batch_size, signal_dimension, symbol_length = GenerateSymbolMat(SymbolFilePath, SymbolMat)
print(QAM_symbol_mat.shape)

lengthRepeat = np.array([1,2,4])
timeRecord = np.zeros((3,))
iterations = 120
batchSize = np.array([1,8,16,32,64])
for BatchSizeIdx in range(5):
    tmp_batch_size = batchSize[BatchSizeIdx]
    tmp_symbol_batch = QAM_symbol_mat[:tmp_batch_size,:,:]
    for SymLenIdx in range(3):
        repeatFactor = lengthRepeat[SymLenIdx]
        tmp_symbol_slice = np.repeat(tmp_symbol_batch,repeatFactor,axis=2)
        tmp_ONNX_input = {QAM_RRC_session.get_inputs()[0].name: tmp_symbol_slice}
        tmpRunningTime = np.zeros((iterations,))
        for iterIdx in range(iterations):
            batch_start = time.time()
            start = time.time()
            QAM_outputs = QAM_RRC_session.run(None, tmp_ONNX_input)
            batch_end = time.time()
            tmpRunningTime[iterIdx] = (batch_end-batch_start)*1000

        timeRecord[SymLenIdx] = np.average(tmpRunningTime[20:])
    print("Current batch size is",tmp_batch_size, "running time of different length are", timeRecord)

# QAM_Mod_time_avg = np.average(timerecord[20:])
# QAM_Mod_time_std = np.std(timerecord[20:])
# print("QAM: inference time Average  is", QAM_Mod_time_avg, 'ms')
# print("QAM: inference time Standard derivation is", QAM_Mod_time_std, 'ms')
