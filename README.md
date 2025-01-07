# belatedMetronome

A highly accurate video beat correction system.  


This program provides both an automatic mode (using midi for alignment) and a manual mode (via GUI) for adjusting the timing of notes in performance videos. Users can fine-tune the alignment by stretching or contracting the timing of both audio and video to match a reference.

Automatic Mode:
A double-loop sequence matching and Dynamic Time Warping (DTW) are used to align performance notes to score notes, while recording the relative time offset coefficients and performance video cut points. 
The relative time coefficients are then applied directly to stretch or compress the corresponding video and audio segments, followed by smooth stitching.  

Manual Mode: 
Offers a graphical interface for manually editing the timing, pitch, and duration of notes.
Recommended for complex sheet music or where precise customization is required.



基于crepe notes的视频节拍纠正系统。
自动模式下使用双循环序列匹配和DTW来匹配演奏音符至乐谱音符，并记录相对时值偏移系数和演奏视频切割点。
应用相对时值系数直接伸缩对应的视频和音频段，然后平滑拼接。

如果乐谱较为复杂或需要自定义时值，则推荐使用手动模式。  









## 未修正的演奏音频：

<img width="1280" alt="Screen Shot 2025-01-06 at 1 42 48 PM" src="https://github.com/user-attachments/assets/78d321e0-4f38-4c31-967a-b443cb842179" />  





## 转换后的演奏midi：

<img width="847" alt="Screen Shot 2025-01-06 at 1 41 26 PM" src="https://github.com/user-attachments/assets/12ab812f-b594-4279-b727-6daab2dbf3e5" />







## 参考midi：

<img width="933" alt="Screen Shot 2025-01-06 at 5 18 20 PM" src="https://github.com/user-attachments/assets/96c07468-ba89-41b4-8edc-2eef09890088" />  


  
演奏音符列表和参考音符列表中，每个音符被表示为：(i, pitch, relative_duration, relative_position)


    


## 计算dtw距离得分，匹配演奏音符和参考音符：

<img width="1280" alt="Screen Shot 2025-01-06 at 4 06 55 PM" src="https://github.com/user-attachments/assets/7bb5719e-53b9-4e0c-9e16-e966b02e09c5" />  


<img width="1280" alt="Screen Shot 2025-01-06 at 4 07 04 PM" src="https://github.com/user-attachments/assets/1f592a2c-d1de-4b3d-a8bd-0f36b95ff078" />  

  
实际计算中考虑了全部音符属性
```python
context_score = dtw_context_similarity(perf_idx, ref_idx)
```

## 匹配结果列表
额外记录了演奏音符的起始时间以切割原演奏视频。
```python
            matches.append({
                "order": perf_order,
                "performance_note": {
                    "start": performance_notes[perf_idx][0],
                    "pitch": performance_notes[perf_idx][1],
                    "duration": performance_notes[perf_idx][2]
                },
                "reference_note": {
                    "start": reference_notes[ref_idx][0],
                    "pitch": reference_notes[ref_idx][1],
                    "duration": reference_notes[ref_idx][2]
                },
                "original_relative_position": perf_position,
                "corrected_relative_position": ref_position,
                "time_correction": time_correction,
                "relative_offset": relative_offset,
                "match_round": 1,
                "dtw_score": dtw_score
            })
```


## 计算 time correction：
ref_duration和perf_duration是归一化到节拍的相对值。
```python
time_correction = ref_duration / perf_duration if perf_duration > 0 else 1.0
```




计算 adjusted time correction（补偿交叉渐变造成的时长损失，crossfade_duration默认为10ms）：
```python
adjusted_time_correction = time_correction * (1 + crossfade_duration / duration)
```




应用 adjusted time correction
```python
corrected_audio = np.stack([
    librosa.effects.time_stretch(segment_audio[0], rate=1 / adjusted_time_correction),
    librosa.effects.time_stretch(segment_audio[1], rate=1 / adjusted_time_correction)
]).astype('float32')
```






## 修正后的演奏音频。稳态高能段（音符）的相对时值和参考midi完全一致：

<img width="1280" alt="Screen Shot 2025-01-06 at 1 43 30 PM" src="https://github.com/user-attachments/assets/9fa99a15-a156-4516-9aa8-16eccd5524d9" />


