"""
From module creator:

    *This code uses an undocumented part of the YouTube API, which is called by the YouTube web-client.
    So there is no guarantee that it won't stop working tomorrow, if they change how things work. I will
    however do my best to make things working again as soon as possible if that happens. So if it stops
    working, let me know!*

"""

from youtube_transcript_api import YouTubeTranscriptApi
import sys

video_id = '5xb6uWLtCsI'

def transcript(video_id):
    t = YouTubeTranscriptApi.get_transcript(video_id)
    script = " ".join([line['text'] for line in t])
    return script

if __name__ == "__main__":
    video_id = '5xb6uWLtCsI'
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
    print(transcript(video_id))

