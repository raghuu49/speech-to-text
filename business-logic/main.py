from faster_whisper import WhisperModel

model=WhisperModel("tiny",device="cpu",compute_type="int8")
segments,info=model.transcribe("../input.mp3", beam_size=1)

with open("../output.txt","w",encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text.strip()+ "\n")
        
print('Audio transcription done!!')
