from enum import Enum
from pathlib import Path
import requests
from fake_useragent import UserAgent
from selectolax.lexbor import LexborHTMLParser
from time import sleep
import re
from tqdm import tqdm

CSV_HEADER = "name; author; tag; tag_coefficient; rating; url\n"
STATUS_OK = 200


class Delay(Enum):
    RATE_LIMIT_DELAY = 1
    MINIMUM_DELAY_BETWEEN_REQUESTS = 0.5


class HtmlPath(Enum):
    TO_NAME = r'#work-names-unit > h2 > span'
    TO_AUTOR = r'#work-names-unit > span > a'
    TO_TAG = r'#workclassif > div.agraylinks > ul'
    TO_RATING = r'#work-rating-unit > div.rating-block-body > dl'


class PathToFile(Enum):
    TOP_LINK = Path("top_link.txt")
    BOOK = Path("book.csv")


useragent = UserAgent()


def main() -> None:
    # Check if the file exists in the directory + Completes the header of our csv file if the file is missing.
    if not PathToFile.BOOK.value.exists():
        create_file_with_header(PathToFile.BOOK.value)

    with open(PathToFile.BOOK.value, "a", encoding='utf-8') as file:
        sites = read_link_from_file(PathToFile.TOP_LINK.value)

        for url in tqdm(sites):
            html, status_code = get_html(url)
            if status_code_checker(status_code, url):
                continue
            file.write(str(text_from_html(html, url)) + '\n')
            sleep(Delay.MINIMUM_DELAY_BETWEEN_REQUESTS.value)


# Create header for csv
def create_file_with_header(path: HtmlPath) -> None:
    with open(path, "w") as fd:
        fd.write(CSV_HEADER)


# Read links to sites from top_link.txt
def read_link_from_file(path: PathToFile) -> list:
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    book_id = [int(element.strip("'{}")) for element in text.split(", ")]
    return [f"https://fantlab.ru/work{i}" for i in sorted(book_id)]


# Getting the html page and the status of the server's response to the request.
def get_html(url: str) -> list:
    headers = {"Accept": "*/*", "User-Agent": useragent.random}
    # Establish a permanent connection
    session = requests.Session()
    session.headers = headers
    adapter = requests.adapters.HTTPAdapter(pool_connections=100,
                                            pool_maxsize=100)
    session.mount('http://', adapter)
    resp = requests.get(url, headers=headers)
    html = resp.text
    return [html, resp.status_code]


def status_code_checker(status_code: int, url: str) -> bool:
    if status_code != STATUS_OK:
        print(f'ERROR_{status_code}:{url}')
        sleep(Delay.RATE_LIMIT_DELAY.value)
        return True
    else:
        return False


# Getting the text genre, author, genres, the degree of belonging of the book to the genre.
def text_from_html(html: str, url: str) -> str:
    tree = LexborHTMLParser(html)
    tree_name = tree.css_first(HtmlPath.TO_NAME.value)
    tree_autor = tree.css_first(HtmlPath.TO_AUTOR.value)
    tree_tag = tree.css_first(HtmlPath.TO_TAG.value)
    tree_rating = tree.css_first(HtmlPath.TO_RATING.value)

    # Most books that lack this element do not have a genre classification.
    if tree_tag is not None:
        element_name = str(tree_name.text())
        element_autor = str(tree_autor.text())
        element_tag = str(tag_name(tree_tag.css("li")))
        element_tag_num = str(tag_num(tree_tag.css("span")))
        element_rating = str(tree_rating.text()).replace("\n", "")

        element_mult = f"{element_name};{element_autor};{element_tag};{element_tag_num};{element_rating};{url}"
        return element_mult


# Book genres
def tag_name(elems: list) -> list:
    check = []

    for li in elems:
        check.append(li.text().split(":"))

    return check


# Degree of belonging of the book to the genre.
def tag_num(elems: list) -> list:
    check = []

    for span in elems:
        html_el = span.html
        if re.search(r"wg[-| ]", html_el) is not None:
            text_from_class = re.findall('"(.*?)"', html_el)
            if len(text_from_class) > 2:
                tag_num(span.css("span"))
            else:
                check.append(text_from_class)
    return check


if __name__ == '__main__':
    main()
