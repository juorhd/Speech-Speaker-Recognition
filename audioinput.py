import pyaudio
import wave
import os
import os.path
import struct
import threading as thread
import numpy as np
import test_speaker
import _pickle as cPickle
import test_speaker
import webrtcvad


CHUNK = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS =5
WAVE_OUTPUT_FILENAME = "output.wav"
save_path =  "speakersample/"

num=0


modelpath = "C:/Users/dingj/Desktop/speaker_recog/speaker_models\\"





gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname
              in gmm_files]

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def detect_voice(frames,vad):
    tmp_frames=[]

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes,16000)
        tmp_frames.append(is_speech)
   # print (tmp_frames)
    num_voiced=tmp_frames.count(True)/len(tmp_frames)
    if (num_voiced > 0.7):
        print ("somebody is talking")
        return 1
    else:
        print ("nobody is talking")
        return 0







def frame_generator(frame_duration_ms, audio):
    n = int(16000 * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / 16000) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)

        timestamp += duration
        offset += n


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
new = WAVE_OUTPUT_FILENAME[:6] + str(num) + WAVE_OUTPUT_FILENAME[6:]
num =num+1
new_frames=[]
vad = webrtcvad.Vad(2)
'''
when we detected voice in the past 1 second , last_state =1
when there is no voice in the past 1 second , last_state =0
'''
last_state=0
new_frames=[]
print("* recording")

while (1):
    '''
    for j in range(0, int(RATE / CHUNK * 1)):
        data = stream.read(CHUNK)
        frames.extend(data)
        new_frames = np.concatenate((new_frames, np.frombuffer(data, dtype=np.int16)))
    '''
    data=stream.read(CHUNK)
    frame_list=frame_generator(30,data)
    frame_list=list(frame_list)
    if (detect_voice(frame_list,vad)==1):
        last_state=1
        new_frames = np.concatenate((new_frames, np.frombuffer(data, dtype=np.int16)))
    elif (detect_voice(frame_list,vad)==0):
        if(last_state==1):
            t1 = thread.Thread(target=test_speaker.test_speaker_, args=(models, speakers, new_frames))
            t1.start()
            new_frames = []
        last_state = 0



print("* done recording")


stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(save_path+new, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(new_frames))
wf.close()
