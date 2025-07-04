import ChatTTS
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

def play_chattts_audio(wav_array):
    """Play ChatTTS audio array using pydub"""
    print(wav_array)
    # Convert to 16-bit PCM
    audio_int16 = (wav_array * 32767).astype(np.int16)
    
    # Create AudioSegment (ChatTTS uses 24kHz)
    audio = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=24000,
        sample_width=2,
        channels=1
    )
    play(audio)

def test_chattts():
    """Test ChatTTS basic functionality"""
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    
    texts = ["Hello! This is ChatTTS speaking.", "How are you today?"]
    wavs = chat.infer(texts)
    
    for i, wav in enumerate(wavs):
        print(f"Playing audio {i+1}")
        play_chattts_audio(wav)

def test_chattts_advanced():
    """Test ChatTTS with voice control"""
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    
    # Sample random speaker
    rand_spk = chat.sample_random_speaker()
    
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )
    
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )
    
    text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
    wavs = chat.infer(text, skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code)
    
    play_chattts_audio(wavs[0])

if __name__ == "__main__":
    test_chattts()
