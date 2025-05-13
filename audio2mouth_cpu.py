import os
import json
import time
import numpy as np
import onnxruntime
import librosa
from extract_paraformer_feature import extract_para_feature
from scipy import signal

class Audio2Mouth(object):
    def __init__(self, use_gpu):

        self.p_list = [str(ii) for ii in range(32)]

        model_path = './weights/model_1.onnx'
        provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
        self.audio2mouth_model=onnxruntime.InferenceSession(model_path, providers=[provider])
        self.w = np.array([1.0]).astype(np.float32)
        self.sp = np.array([2]).astype(np.int64)


    def butter_lowpass_filtfilt(self, data, order=4, cutoff=7, fs=30):
        wn = 2 * cutoff / fs
        b, a = signal.butter(order, wn, 'lowpass', analog=False)
        output = signal.filtfilt(b, a, data, padtype=None, axis=0)
        return output  

    def mouth_smooth(self, param_res):
        for key in param_res[0]:
            val_list = []
            for ii in range(len(param_res)):
                val_list.append(param_res[ii][key])
            val_list = self.butter_lowpass_filtfilt(np.asarray(val_list), order=4, cutoff=10, fs=30)
            for ii in range(len(param_res)):
                param_res[ii][key] = val_list[ii]
        return param_res

    def geneHeadInfo(self, sampleRate, bits, sampleNum):
        import struct
        rHeadInfo = b'\x52\x49\x46\x46'
        fileLength = struct.pack('i', sampleNum + 36)
        rHeadInfo += fileLength
        rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'
        rHeadInfo += struct.pack('i', sampleRate)
        rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))
        rHeadInfo += b'\x02\x00'
        rHeadInfo += struct.pack('H', bits)
        rHeadInfo += b'\x64\x61\x74\x61'
        rHeadInfo += struct.pack('i', sampleNum)
        return rHeadInfo
    
    def inference(self, subtitles=None, input_audio=None, sample_rate=16000):
        f1 = time.time()

        frame_cnt = int(len(input_audio) / sample_rate * 30)
        au_data = extract_para_feature(input_audio, frame_cnt)
        ph_data = np.zeros((au_data.shape[0], 2))
        
        audio_length = ph_data.shape[0] / 30
        print('extract all feature in {}s'.format(round(time.time() - f1, 3)))
        print('audio length: {}s'.format(round(audio_length, 3)))
            
        param_res = []
        interval = 1.0
        frag = int(interval * 30 / 5 + 0.5)
        
        start_time = 0.0
        end_time = start_time + interval
        is_end = False
        while True:
            # 计算当前时间段的起始和结束帧
            start = int(start_time * sample_rate)
            end = start + sample_rate
            # 检查是否已到达音频的结尾
            if end_time >= audio_length:
                is_end = True
                # 如果已到达结尾，则调整起始和结束帧以确保处理最后一段音频
                end = int(audio_length * sample_rate)
                start = end - sample_rate
                start_time = audio_length - interval
                end_time = audio_length
            # 计算当前时间段的起始和结束帧在30fps下的对应帧数
            start_frame = int(start_time * int(30))
            end_frame = start_frame + int(30 * interval)

            # 提取当前时间段的音频和唇部运动特征
            input_au = au_data[start_frame:end_frame]
            input_ph = ph_data[start_frame:end_frame]
            # 将特征转换为模型所需的格式
            input_au = input_au[np.newaxis,:].astype(np.float32)
            input_ph = input_ph[np.newaxis,:].astype(np.float32)
            
            # 使用模型进行推理，得到输出和特征
            output, feat = self.audio2mouth_model.run(['output', 'feat'],{'input_au':input_au,'input_ph':input_ph,'input_sp':self.sp,'w':self.w})
            
            # 根据当前时间段的位置处理输出
            if start_time == 0.0:
                if is_end is False:
                    # 如果当前时间段在音频的开头且不是最后一段，则处理输出
                    for tt in range(int(30 * interval) - frag):
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0,tt,ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
                else:
                    # 如果当前时间段在音频的开头且是最后一段，则处理输出
                    for tt in range(int(30 * interval)):
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0,tt,ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
            elif is_end is False:
                # 如果当前时间段不是音频的开头或结尾，则处理输出
                for tt in range(frag,int(30 * interval) - frag):
                    frame_id = start_frame + tt
                    if frame_id < len(param_res):
                        scale = min((len(param_res) - frame_id) / frag, 1.0)
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0,tt,ii])
                            value = (1 - scale) * value + scale * param_res[frame_id][key]
                            param_res[frame_id][key] = round(value, 3)
                    else:
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0,tt,ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
            else:
                # 如果当前时间段在音频的结尾，则处理输出
                for tt in range(frag,int(30 * interval)):
                    frame_id = start_frame + tt
                    if frame_id < len(param_res):
                        scale = min((len(param_res) - frame_id) / frag, 1.0)
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0,tt,ii])
                            value = (1 - scale) * value + scale * param_res[frame_id][key]
                            param_res[frame_id][key] = round(value, 3)
                    else:
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0,tt,ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
            
            # 更新当前时间段的起始和结束时间
            start_time = end_time - (frag / 10)
            end_time = start_time + interval
            # 检查是否已到达音频的结尾
            if is_end is True:
                break

        param_res = self.mouth_smooth(param_res)
        
        print("generate {} frames in {}s with avg inference time {}ms/frame".format(len(param_res), round(time.time() - f1, 3), round((time.time() - f1) / len(param_res) * 1000, 3)))
        
        return param_res, None, None
