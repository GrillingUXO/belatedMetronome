# belatedMetronome
Belated Metronome
A highly highly accurate video beat correction system based on Crepe notes.

This program provides both an automatic mode (using midi for alignment) and a manual mode (via GUI) for adjusting the timing of notes in performance videos. Users can fine-tune the alignment by stretching or contracting the timing of both audio and video to match a reference.

Automatic Mode:
Automatically corrects timing using a double-loop Dynamic Time Warping (DTW) algorithm to align performance notes with reference sheet music.
Computes relative time deviation coefficients and uses these to stretch or contract video and audio in alignment with the reference.
Extracts note information using Crepe and cuts audio/video based on note boundaries.


基于crepe notes的视频节拍纠正系统。
程序提供用户界面以手动编辑演奏视频中音符的时值，也可以上传原曲的musicXML来自动纠正。

自动模式下使用双循环DTW来匹配演奏音符至乐谱音符，并记录相对时值偏移系数。
使用crepe notes提供的音符列表来切割音频和视频，同时应用相对时值系数以直接伸缩视频和音频。

如果乐谱较为复杂或需要自定义时值，则推荐使用手动模式。




未修正的演奏音频。

<img width="1280" alt="Screen Shot 2025-01-06 at 1 42 48 PM" src="https://github.com/user-attachments/assets/78d321e0-4f38-4c31-967a-b443cb842179" />




dtw匹配示例

<img width="1280" alt="Screen Shot 2025-01-06 at 4 06 55 PM" src="https://github.com/user-attachments/assets/7bb5719e-53b9-4e0c-9e16-e966b02e09c5" />

<img width="1280" alt="Screen Shot 2025-01-06 at 4 07 04 PM" src="https://github.com/user-attachments/assets/1f592a2c-d1de-4b3d-a8bd-0f36b95ff078" />




修正后的演奏音频。稳态高能段（音符）的相对时值和参考midi完全一致

<img width="1280" alt="Screen Shot 2025-01-06 at 1 43 30 PM" src="https://github.com/user-attachments/assets/9fa99a15-a156-4516-9aa8-16eccd5524d9" />


