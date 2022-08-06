import requests
from bs4 import BeautifulSoup

def query(q):
    URL = f"https://www.google.com/search?q={q}&hl=en"

    header = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
        'Accept-Language':'en-US'
    }

    page = requests.get(URL, headers=header)
    soup = BeautifulSoup(page.content, 'html.parser')
    if "weather" in q:
        result = soup.find(class_ = 'wob_t q8U8x').get_text()
        print("It is "+result+" degree celcius.")
    elif "flip" in q:
        result = soup.find(class_ = 'PmF7Ce').get_text()
        print("You got "+result+"!")
    else:
        result = soup.find(class_ = 'vk_bk dDoNo FzvWSb').get_text()
        print(result)


while True:
    try:
        q = input("Ask me sth: ")
        query(q)
    except Exception:
        print("Sorry, I didn't get that.")
    user_input = input("Press y to continue: ")
    if user_input != 'y':
        break

# QUERIES:
# whats the date tomorrow
# flip a coin
# whats the weather today