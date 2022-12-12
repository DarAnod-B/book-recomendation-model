from enum import Enum
import requests
from selectolax.lexbor import LexborHTMLParser
import argparse


class Link(Enum):
    url_best_book = r"https://fantlab.ru/rating/work/best?all=1&type=1&threshold=50"
    url_popular_book = r"https://fantlab.ru/rating/work/popular?all=1&type=1&threshold=50"
    url_titled_book = r"https://fantlab.ru/rating/work/titled?all=1&type=1"


def main() -> None:
    with open(r'top_link.txt', 'w', encoding="utf-8") as f:
        f.write(
            str(
                set(
                    text_from_html(get_html(Link.url_best_book.value)) +
                    text_from_html(get_html(Link.url_popular_book.value)) +
                    text_from_html(get_html(Link.url_titled_book.value)))))


def text_from_html(html: str) -> list:
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


def get_html(url: str) -> str:
    with requests.Session() as session:
        args = parse_args()

        #  python .\top_book.py --login $(cat .\secrets_data\login) --password $(cat .\secrets_data\password)
        payload = {'login': args.login, 'password': args.password}
        login_url = r"https://fantlab.ru/login"

        session.post(login_url, data=payload)
        result = session.get(url)
        html = result.text
        return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--login', type=str, required=True)
    parser.add_argument('-p', '--password', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
