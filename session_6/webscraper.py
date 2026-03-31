import requests

from bs4 import BeautifulSoup

# Send a GET request to the URL

response = requests.get('https://chatgpt.com/')

# Parse the HTML content using BeautifulSoup

soup = BeautifulSoup(response.text, 'html.parser')

#print(f"soup : {soup}")
# Find the first <h1> tag

first_header = soup.find('h1')

print('First <h1> tag text:', first_header)
# Find all <a> tags (links)

all_links = soup.find_all('a')

print('All <a> tag hrefs:')

for link in all_links:
    print(link.get('href'))

# Access attributes of an element

if len(all_links) >0 :
    first_link = all_links[0]
    print('First link text:', first_link.text)
    print('First link href:', first_link.get('href'))
# Navigate using parent and siblings

if first_header:
    parent_element = first_header.parent
    print('Parent of first <h1> tag:', parent_element.name)

