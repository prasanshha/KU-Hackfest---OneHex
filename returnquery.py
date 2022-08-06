import requests
from bs4 import BeautifulSoup

def query(q):
    print(q)
    URL = f"https://www.google.com/search?q={q}&hl=en"

    header = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
        'Accept-Language':'en-US'
    }

    page = requests.get(URL, headers=header)
    soup = BeautifulSoup(page.content, 'html.parser')
    response = ""
    if "weather" in q:
        result = soup.find(class_ = 'wob_t q8U8x').get_text()
        response = "It will be "+result+" degree celcius."
    elif "flip" in q:
        result = soup.find(class_ = 'PmF7Ce').get_text()
        response = "You got "+result+"!"
    elif "time" in q:
        result = soup.find(class_ = 'gsrt vk_bk FzvWSb YwPhnf').get_text()
        response = result
    else:
        result = soup.find(class_ = 'LC20lb MBeuO DKV0Md').get_text()
        response = "Here's what I found:\n"+result

    return response




# while True:
#     try:
#         q = input("Ask me sth: ")
#         query(q)
#     except Exception:
#         print("Sorry, I didn't get that.")
#     user_input = input("Press y to continue: ")
#     if user_input != 'y':
#         break

# QUERIES:
# whats the time now
# flip a coin
# whats the weather tomorrow