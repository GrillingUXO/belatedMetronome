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
使用crepe note提供的音符列表来切割音频和视频，同时应用相对时值系数以直接伸缩视频和音频。

如果乐谱较为复杂或需要自定义时值，则推荐使用手动模式。




