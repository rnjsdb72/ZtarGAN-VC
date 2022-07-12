import pandas as pd
from tqdm import tqdm
from glob import glob
import os

from pytube import Channel, Playlist

import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome

def download_mp3(channel):
    c = Channel(channel)

    df_lst = []
    print('Start Download')
    videos = c.videos
    for video in tqdm(videos):
        error = 0
        try:
            video.streams.filter(only_audio=True).first().download('./download/')
        except:
            error = 1
        url = video.watch_url
        title = video.title
        length = video.length
        df_lst.append(pd.DataFrame({'url':url, 'title':title, 'length':length, 'error':error}, index=[0]))
    print('Download Complete!')
    df = pd.concat(df_lst, axis=0).reset_index(drop=True)
    df.to_csv(f'./download/df_music.csv', index=False)
    print(f'Download "df_music.csv" Success!')

def convert():
    print('Start Converting')
    files = glob('./download/*.mp4')
    for file in tqdm(files):
        if not os.path.isdir(file):
            filename = os.path.splitext(file)
            try:
                os.rename(file, filename[0]+'.mp3')
            except:
                pass
    print('Convert Success!')

def crawling_lyrics():
    df = pd.read_csv('./download/df_music.csv')

    print('\tPreprocessing...')

    pl_1 = list(Playlist('https://www.youtube.com/playlist?list=PL5rkMpxC5Ex9T89yMwWTTl38OF6NMxrT7'))
    pl_2 = list(Playlist('https://www.youtube.com/watch?v=IPtyDLwMyJg&list=PL5rkMpxC5Ex9SZ5fbVBm0WhlA48J5C-He'))
    pl_3 = list(Playlist('https://www.youtube.com/watch?v=22qlOevg4KY&list=PL5rkMpxC5Ex_Dyg-PGwKGPSjZ80tHQC3l'))

    except_lst = pl_1 + pl_2 + pl_3

    df = df.query('url not in @except_lst').reset_index(drop=True)
    df['title2'] = df.title.map(lambda x: x.split(']')[1].split('/')[0].split('TJ')[0].strip())

    dir_driver = chromedriver_autoinstaller.install()
    browser = Chrome(dir_driver)
    browser.implicitly_wait(15)

    print('\tCrawling Lyrics...')
    lyrics = []
    error = 0
    for keyword in tqdm(df.title):
        try:
            browser.get(f'https://www.melon.com/search/lyric/index.htm?q={keyword}')
            browser.find_element(By.CSS_SELECTOR, '.cntt_lyric a').click()
            browser.find_element(By.CSS_SELECTOR, 'button.button_more.arrow_d').click()
            lyric = '. '.join(browser.find_element(By.CSS_SELECTOR, 'div#d_video_summary.lyric.on').text.split('\n'))
            lyrics.append(lyric)
        except:
            lyrics.append('')
            error += 1
    df['lyrics'] = lyrics
    df.to_csv(f'./download/df_music.csv', index=False)
    print(f'\tDownload "df_music.csv" Success!')
    print(f'Error Rate:\t {error}/{df.shape[0]}')

if __name__ == '__main__':
    url = input('Enter the URL: \t')
    download_mp3(url)
    convert()
    print('Ready to Crawling Lyrics...')
    crawling_lyrics()

