from pathlib import Path
import requests
from fake_useragent import UserAgent
from selectolax.lexbor import LexborHTMLParser
from time import sleep
from tqdm import tqdm
from enum import Enum

CSV_HEADER = "book; user; grade; rating_grade; publication_date; comment \n"
STATUS_OK = 200


class PathToFile(Enum):
    TOP_LINK = Path("top_link.txt")
    USER_CSV = Path("user.csv")


class Delay(Enum):
    RATE_LIMIT_DELAY = 1
    MINIMUM_DELAY_BETWEEN_REQUESTS = 0.5


useragent = UserAgent()


def main() -> None:
    # Checking if the file exists in the directory + adding the header of our csv file if the file is missing.
    if not PathToFile.USER_CSV.value.exists():
        create_file_with_header(PathToFile.USER_CSV.value)

    with open(PathToFile.USER_CSV.value, "a", encoding='utf-8') as file:
        sites = read_link_from_file(PathToFile.TOP_LINK.value)

        for url in tqdm(sites):
            html, _ = get_html(url)

            # Parsing a list with reviews
            line = ''.join(score_user(score_link(html, url)))

            # Check if the string is empty
            if line:
                file.write(line)


# Create header for csv
def create_file_with_header(path: PathToFile) -> None:
    with open(path, "w") as df:
        df.write(CSV_HEADER)

# Read links to sites from top_link.txt


def read_link_from_file(path: PathToFile) -> list:
    with open(path, 'r', encoding="utf-8") as file:
        text = file.read()
    print(text)
    book_id = [int(element.strip("'{}")) for element in text.split(", ")]
    return [f"https://fantlab.ru/work{i}" for i in sorted(book_id)]


# Get the html page and request response status.
def get_html(url: str) -> tuple[str, int]:
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


# Get links to review pages
def score_link(html: str, url: str) -> list:
    tree = LexborHTMLParser(html)
    tree_users = tree.css_first(r'span.page-links')

    link_list = []
    if tree_users is not None:
        tree_users_list = tree_users.css(r'a')
        for user in tree_users_list:
            # Link to comment page
            link = url + user.attributes['href']
            link_list.append(link)
        return link_list
    else:
        link_list.append(url)
        return link_list


# Get user feedback
def score_user(links: list) -> list:
    score_list = []

    # Follow links to review pages
    for url in links:
        sleep(Delay.MINIMUM_DELAY_BETWEEN_REQUESTS.value)
        html, status_code = get_html(url)
        tree = LexborHTMLParser(html)

        if status_code != STATUS_OK:
            print(f'ERROE_{status_code}:{url}')
            sleep(Delay.RATE_LIMIT_DELAY.value)
            return

        score = tree.css("div.responses-list > div.response-item")
        if score is not None:
            for user in score:
                # book; user; book_rating; rating_review; publication_date; review
                book_link = url.split('?')[0]
                user_id = user.css_first(
                    r'p.response-autor-info>b>a').attributes['href']
                book_rating = user.css_first(
                    r'div.clearfix>div.response-autor-mark>b>span').text()
                comment_rating = user.css_first(
                    r'div.response-votetab>span:nth-of-type(2)').text()
                data_score = user.css_first(
                    r'p.response-autor-info>span').attributes['content']
                body_score = user.css_first(
                    r'div.response-body-home').text().replace('\n', ' ')
                score_list.append(
                    f'{book_link};{user_id};{book_rating};{comment_rating};{data_score};{body_score}\n'
                )

    return score_list


if __name__ == '__main__':
    main()
