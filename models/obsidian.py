
"""
This will be a command line utility I can use to add things to my obsidian vault.

Ingests the following data:
- youtube url -> transcript (Morphy/YouTube)
- article url -> article text (Morphy/extracted_articles)

This uses gpt-4o by default, so it's not free. Switch to an ollama model if you want to use it for free.

Next up: automatically grab html (or PDF) versions of arxiv papers and summarize them.
"""

from download_article import download_article
from download_youtube_transcript import download_transcript
from Chain import Chain, Model, Prompt
import sys
import os

# get OBSIDIAN_PATH from environment variable
obsidian_path = os.environ.get('OBSIDIAN_PATH')

# switch this if on different comp
summarized_urls = obsidian_path + '/Summarized_URLs.md'

prompt_string = """
Summarize the key points from the following article or youtube transcript. Structure the summary with clear headings and subheadings.
Distill the main ideas concisely, but include key details and takeaways. Use direct quotes sparingly, only to highlight the most insightful or impactful statements.
Aim for a summary that is around 300-800 words in length.

Structure the summary as follows:

**Main Topic 1**
- Key point
- Key point

**Main Topic 2**
- Key point
- Key point

**Conclusion**
Summarize overarching takeaways and conclusions
What are the main lessons or insights that someone should take away from this text?

At the end of your answer, provide a title surrounded by two square brackets, like this: [[Title of the Article]]
The Title should not contain any special characters, especially not square brackets. Do not include colons.

TRANSCRIPT: 

{{transcript}}
""".strip()

def retrieve_summarized_urls():
    with open(summarized_urls, 'r') as f:
        urls = f.read()
    urls = urls.split('\n')
    return urls

def save_summarized_url(url):
    with open(summarized_urls, 'a') as f:
        f.write(url + '\n')

def parse_input(input):
    urls = retrieve_summarized_urls()
    if input in urls:
        raise ValueError('URL has already been summarized.')
    elif 'youtube' in input:
        return download_transcript(input)
    elif 'http' in input:
        return download_article(input)
    else:
        raise ValueError('Input must be a YouTube URL or an article URL.')

def summarize_text(text):
    model = Model('gpt')
    prompt = Prompt(prompt_string)
    chain = Chain(model=model, prompt=prompt)
    summary = chain.run(text, verbose=False)
    summary = summary.content.strip()
    if '[[' and ']]' not in summary.split('\n')[-1]:
        return ValueError('Model did not generate a title.')
    else:
        title, summary = summary.split('\n')[-1], summary.split('\n')[:-1]
        title = title.replace('[[', '').replace(']]', '')
        summary = '\n'.join(summary)
        return title, summary

def save_to_obsidian(title, summary, url):
    filename = f'{obsidian_path}/{title}.md'
    summary = url + "\n\n" + summary
    with open(filename, 'w') as f:
        f.write(summary)
    save_summarized_url(url)
    return filename

if __name__ == '__main__':
    url = 'https://www.androidauthority.com/rabbit-r1-is-an-android-app-3438805'
    if len(sys.argv) == 1:
        print('No URL provided. Provide a YouTube URL or an article URL.')
        sys.exit(1)
    if len(sys.argv) > 1:
        url = sys.argv[1]
    try:
        data = parse_input(url)
    except ValueError as e:
        print("ValueError: " + str(e))
        sys.exit(1)
    if data:
        title, summary = summarize_text(data)
        print(title)
        print('=============================================')
        print(summary)
        print('=============================================')
        filename = save_to_obsidian(title, summary, url)
        print(f'Saved to Obsidian: {filename}')
    else:
        print('Output is empty.')
