import requests
from bs4 import BeautifulSoup as bs
import re
import time
# from requests_html import HTMLSession

from selenium import webdriver

driver = webdriver.Chrome(executable_path='/Users/thtrieu/chromedriver')


with open('tgtp.webpagedata') as f:
  c = f.read()

l = re.findall('GEO[0-9]*', c)
l = list(set(l))
print(sorted(l))

print(len(l))
# session = HTMLSession()

for i, c in enumerate(l):
  href = "http://hilbert.mat.uc.pt/TGTP/Problems/reportPrb.php?argumento=" + c
  driver.get(href)
  if i < 3:
    time.sleep(5)
  else:
    time.sleep(0.5)
  driver.get(href)
  html = driver.page_source
  # print(soup)
  print(href)

  soup = bs(html)
  print(soup)

  with open('tgtp/{}.html'.format(c), 'w') as f:
    f.write(str(soup))

  # a = raw_input()
  