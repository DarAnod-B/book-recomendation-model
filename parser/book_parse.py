from pathlib import Path
import requests
from fake_useragent import UserAgent
from selectolax.lexbor import LexborHTMLParser
from time import sleep
import re
from tqdm import tqdm

RATE_LIMIT_DELAY = 1
MINIMUM_DELAY_BETWEEN_REQUESTS = 0.5

PATH_TO_NAME = r'#work-names-unit > h2 > span'
PATH_TO_AUTOR = r'#work-names-unit > span > a'
PATH_TO_TAG = r'#workclassif > div.agraylinks > ul'
PATH_TO_RATING = r'#work-rating-unit > div.rating-block-body > dl'
TOP_LINK_PATH = Path("top_link.txt")
BOOK_PATH = Path("book.csv")
CSV_HEADER = "name; author; tag; tag_coefficient; rating; url\n"

useragent = UserAgent()


def main():
    # Проверка файл на наличие в директории + Добавляет заголовок нашего csv файла, если файл отсутствует.
    if not BOOK_PATH.exists():
        create_file_with_header(BOOK_PATH)

    with open(BOOK_PATH, "a", encoding='utf-8') as file:
        sites = read_link_from_file(TOP_LINK_PATH)

        for url in tqdm(sites):
            html, status_code = get_html(url)
            # Проверка ответа сервера
            if status_code_checker(status_code, url):
                continue
            file.write(str(text_from_html(html, url)) + '\n')

            sleep(MINIMUM_DELAY_BETWEEN_REQUESTS)


# Чтение ссылок на сайты из top_link.txt
def read_link_from_file(path: Path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    book_id = [int(element.strip("'{}")) for element in text.split(", ")]
    return [f"https://fantlab.ru/work{i}" for i in sorted(book_id)]


# Получение html страницы и статуса ответа сервера на запрос.
def get_html(url):
    headers = {"Accept": "*/*", "User-Agent": useragent.random}
    # Устанавливаем постоянное соединение
    session = requests.Session()
    session.headers = headers
    adapter = requests.adapters.HTTPAdapter(pool_connections=100,
                                            pool_maxsize=100)
    session.mount('http://', adapter)
    resp = requests.get(url, headers=headers)
    html = resp.text
    return html, resp.status_code


# Жанры книг
def tag_name(list_tags):
    check = []
    # При каждой итерации проходится один блок (li)
    tags = list_tags.css("li")
    for e in tags:
        check.append(e.text().split(":"))

    return check


# Степень принадлежности книги к жанру.
def tag_num(elems):
    check = []
    # При каждой итерации проходится один блок (span)
    for e in elems:
        html_el = e.html
        if re.search(r"wg[-| ]", html_el) is not None:
            text_from_class = re.findall('"(.*?)"', html_el)
            if len(text_from_class) > 2:
                tag_num(e.css("span"))
            else:
                check.append(text_from_class)
    return check


# Получение текста жанр, автор, жанры, степень принадлежности книги к жанру.
def text_from_html(html, url):
    tree = LexborHTMLParser(html)
    tree_name = tree.css_first(PATH_TO_NAME)
    tree_autor = tree.css_first(PATH_TO_AUTOR)
    tree_tag = tree.css_first(PATH_TO_TAG)
    tree_rating = tree.css_first(PATH_TO_RATING)

    # Большинство книг, в которых отсутствует данный элемент, не имеют жанровой классификации.
    if tree_tag is not None:
        element_name = str(tree_name.text())
        element_autor = str(tree_autor.text())
        element_tag = str(tag_name(tree_tag))
        element_tag_num = str(tag_num(tree_tag.css("span")))
        element_rating = str(tree_rating.text()).replace("\n", "")

        element_mult = f"{element_name};{element_autor};{element_tag};{element_tag_num};{element_rating};{url}"
        return element_mult


# Создать заголовок для csv
def create_file_with_header(path: Path) -> None:
    with open(path, "w") as fd:
        fd.write(CSV_HEADER)


def status_code_checker(status_code, url):
    if status_code != 200:
        print(f'ERROR_{status_code}:{url}')
        sleep(RATE_LIMIT_DELAY)
        return True
    else:
        return False


if __name__ == '__main__':
    main()
