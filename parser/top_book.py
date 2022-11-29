import requests
from selectolax.lexbor import LexborHTMLParser

url_1 = r"https://fantlab.ru/rating/work/best?all=1&type=1&threshold=50"
url_2 = r"https://fantlab.ru/rating/work/popular?all=1&type=1&threshold=50"
url_3 = r"https://fantlab.ru/rating/work/titled?all=1&type=1"


def get_html(url2):
    with requests.Session() as session:
        payload = {'login': 'dari', 'password': 'mzMj5tr6nWKtEfd'}
        url = r"https://fantlab.ru/login"

        session.post(url, data=payload)
        result = session.get(url2)
        return result.text


def text_from_html(html):
    tree = LexborHTMLParser(html)
    tree_url = tree.css(
        "body > div.layout > div > div > div.main-container > main > table:nth-child(6) > tbody"
    )

    check = []
    # Большинство книг, в которых отсутствует данный элемент, не имеют жанровой классификации.
    if tree_url is not None:
        for i in tree_url[0].css("a"):
            check.append(i.attrs['href'][5:])
        return check


with open(r'top_link.txt', 'w', encoding="utf-8") as f:
    f.write(
        str(
            set(
                text_from_html(get_html(url_1)) +
                text_from_html(get_html(url_2)) +
                text_from_html(get_html(url_3)))))
