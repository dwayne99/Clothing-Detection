from selenium import webdriver
import urllib.request
import os
import json
from bs4 import BeautifulSoup
from pdb import set_trace
import time
import re
from selenium.webdriver.chrome.options import Options

DATASET_PATH = '../dataset_myntra/'

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome('../../drivers/chromedriver')
query = input('Enter search item: ')
home_link ='https://www.myntra.com/' 

for page in range(11,30):
    
    link = home_link + query + '?p=' + str(page)
    driver.get(link)
    time.sleep(2)

    response = driver.page_source

    soup = BeautifulSoup(response,'html.parser')
    mydivs = soup.findAll("li", {"class": "product-base"})
    for x, product in enumerate(mydivs):
        
        try:
            product_name = product.find('img')['title']
           
            folder_name = DATASET_PATH+product_name
            os.mkdir(folder_name)
            follow_link = home_link + product.find('a')['href']
            driver.get(follow_link)
            time.sleep(2)
            img_soup = BeautifulSoup(driver.page_source,'html.parser')
            imgs = img_soup.findAll('div',{'class':'image-grid-image'})
            for i,img in enumerate(imgs):
                link = re.findall(r'"(.*?)"',img['style'])[0]
                img_path = folder_name + '/'+str(i)+'.jpg'
                #set_trace()
                urllib.request.urlretrieve(link,img_path)

            print(f'Page {page} : Image {x}')
        except:
            pass

